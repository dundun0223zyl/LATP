#!/usr/bin/env python3
"""
Main coordination script for the two-phase web crawler with Gemini evaluation.
This script orchestrates the workflow between the initial scraping phase and the link evaluation phase.
"""

import os
import json
import argparse
import asyncio
from datetime import datetime
from rich.console import Console
from rich.prompt import Confirm
from rich import print as rprint

from level_scraper import LevelScraper
from link_evaluator import LinkEvaluator

class CrawlerCoordinator:
    def __init__(self, start_url, output_folder="crawl_output", gemini_api_key=None, 
                 min_delay=1.5, max_delay=3.0, request_timeout=30):
        """
        Initialize the crawler coordinator.
        
        Args:
            start_url (str): Starting URL for the crawl
            output_folder (str): Base output folder
            gemini_api_key (str): API key for Google's Gemini API
            min_delay (float): Minimum delay between requests
            max_delay (float): Maximum delay between requests
            request_timeout (int): Request timeout in seconds
        """
        self.start_url = start_url
        self.base_output_folder = output_folder
        self.gemini_api_key = gemini_api_key
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.request_timeout = request_timeout
        
        # Create a timestamped folder for this crawl
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(output_folder, f"crawl_{timestamp}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize console for rich output
        self.console = Console()
        
        # Track crawl progress
        self.current_level = 0
        self.max_levels = 0  # Will be set by the user during the crawl
        
        # Global sets for tracking URLs across all levels
        self.all_clean_links = set()  # Links approved for crawling
        self.rejected_links = set()   # Links explicitly rejected
        self.processed_links = set()  # Links that have been processed
        
        # ADDED: Global tracking for normalized URLs to prevent duplicate visits
        self.global_visited_urls = set()  # All visited URLs across all levels
        self.global_normalized_urls = set()  # Normalized versions for efficient checking
        self.global_normalized_base_urls = set()  # Normalized base URLs (without fragments)
        
        # Create a global cache file for Gemini decisions
        self.gemini_cache_file = os.path.join(self.output_folder, "gemini_decisions_cache.json")
        
    async def crawl_level(self, urls, level):
        """
        Crawl a specific level of pages.
        
        Args:
            urls (list): List of URLs to crawl
            level (int): Current crawl level
        
        Returns:
            str: Path to the collected links file
        """
        self.console.print(f"\n[bold green]Starting crawl of level {level}[/bold green]")
        
        # Deduplicate URLs - remove any that have already been processed
        # BUT KEEP the start URL for level 0 (even if it's in processed_links)
        unique_urls = []
        for url in urls:
            # For level 0, always include the seed URL
            if level == 0 and url == self.start_url:
                unique_urls.append(url)
                continue
                
            # For other levels, filter out already processed URLs
            if url not in self.processed_links:
                unique_urls.append(url)
                self.processed_links.add(url)
        
        self.console.print(f"URLs to crawl: {len(urls)} (deduplicated to {len(unique_urls)})")
        
        # Create level-specific output folder
        level_output = os.path.join(self.output_folder, f"level_{level}")
        os.makedirs(os.path.join(level_output, "text_content"), exist_ok=True)
        os.makedirs(os.path.join(level_output, "downloaded_files"), exist_ok=True)
        
        # Initialize the scraper for this level - WITH GLOBAL URL TRACKING
        scraper = LevelScraper(
            seed_urls=unique_urls,
            max_depth=0,  # Important: Set to 0 to only crawl the URLs provided without following links
            output_folder=level_output,
            min_delay=self.min_delay,
            max_delay=self.max_delay,
            request_timeout=self.request_timeout,
            # ADDED: Pass global tracking sets to maintain state between levels
            visited_urls=self.global_visited_urls,
            normalized_visited_urls=self.global_normalized_urls,
            normalized_base_urls=self.global_normalized_base_urls
        )
        
        # IMPORTANT: Modify the scraper to use the level folders directly
        # This prevents creating nested level_0 folders
        scraper.use_direct_folder_structure = True
        
        # Perform the crawl
        collected_links = await scraper.crawl()
        
        # ADDED: Update global tracking with the scraper's visited URLs
        self.global_visited_urls.update(scraper.visited_urls)
        self.global_normalized_urls.update(scraper.normalized_visited_urls)
        self.global_normalized_base_urls.update(scraper.normalized_base_urls)
        
        # Save the links to a level-specific file
        links_file = os.path.join(level_output, "collected_links.json")
        with open(links_file, "w", encoding="utf-8") as f:
            json.dump(collected_links, f, indent=2)
        
        # ADDED: Save global tracking info for debugging
        tracking_info = {
            "level": level,
            "global_visited_count": len(self.global_visited_urls),
            "global_normalized_count": len(self.global_normalized_urls),
            "global_normalized_base_count": len(self.global_normalized_base_urls)
        }
        
        tracking_file = os.path.join(level_output, "url_tracking_info.json")
        with open(tracking_file, "w", encoding="utf-8") as f:
            json.dump(tracking_info, f, indent=2)
        
        self.console.print(f"[bold green]Completed crawl of level {level}[/bold green]")
        self.console.print(f"Collected {len(collected_links)} pages of links")
        self.console.print(f"Links saved to {links_file}")
        self.console.print(f"Global tracking: {len(self.global_visited_urls)} URLs visited across all levels")
        
        return links_file
    
    def evaluate_links(self, links_file, level):
        """
        Evaluate links using Gemini and manual review.
        
        Args:
            links_file (str): Path to the JSON file with collected links
            level (int): Current crawl level
        
        Returns:
            list: Clean list of links for the next level
        """
        self.console.print(f"\n[bold blue]Evaluating links from level {level}[/bold blue]")
        
        if not self.gemini_api_key:
            self.console.print("[bold yellow]No Gemini API key provided. Skipping AI evaluation.[/bold yellow]")
            
            # Load the links directly without Gemini filtering
            with open(links_file, "r", encoding="utf-8") as f:
                links_data = json.load(f)
            
            # Extract all internal links
            all_internal_links = []
            for page in links_data:
                for link in page.get("internal_links", []):
                    # Skip already processed or rejected links
                    if link in self.processed_links or link in self.rejected_links:
                        continue
                    if link not in all_internal_links:
                        all_internal_links.append(link)
            
            # Save to a clean links file
            clean_links_file = os.path.join(os.path.dirname(links_file), "clean_links.json")
            with open(clean_links_file, "w", encoding="utf-8") as f:
                json.dump({
                    "clean_links": all_internal_links,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_count": len(all_internal_links)
                }, f, indent=2)
            
            self.console.print(f"[bold yellow]No Gemini filtering applied. All {len(all_internal_links)} internal links saved.[/bold yellow]")
            
            # Allow manual review
            if Confirm.ask("Would you like to manually review the links?"):
                # Implement simple manual review here
                self.console.print("[bold]Manual review not implemented yet. Please edit the file directly:[/bold]")
                self.console.print(clean_links_file)
            
            # Add these to our global tracking
            self.all_clean_links.update(all_internal_links)
            
            return all_internal_links
        
        # Use Gemini for evaluation
        clean_links_file = os.path.join(os.path.dirname(links_file), "clean_links.json")
        evaluator = LinkEvaluator(
            links_file=links_file,
            gemini_api_key=self.gemini_api_key,
            output_file=clean_links_file,
            decision_cache_file=self.gemini_cache_file
        )
        
        # Process the links
        evaluator.process_all_pages()
        
        # Show Gemini's analysis
        evaluator.display_gemini_analysis()
        
        # Allow manual review and editing
        clean_links = evaluator.manually_review_links()
        
        # Update our global tracking sets using LinkEvaluator's data
        if hasattr(evaluator, 'all_keep_urls'):
            self.all_clean_links.update(evaluator.all_keep_urls)
        if hasattr(evaluator, 'all_reject_urls'):
            self.rejected_links.update(evaluator.all_reject_urls)
        
        self.console.print(f"[bold blue]Link evaluation complete. {len(clean_links)} clean links ready for next level.[/bold blue]")
        return clean_links
    
    async def run_workflow(self):
        """Run the complete crawling workflow with explicit stops between levels."""
        self.console.print(f"[bold]Starting Two-Phase Web Crawler[/bold]")
        self.console.print(f"Starting URL: {self.start_url}")
        self.console.print(f"Output folder: {self.output_folder}")
        
        # Ask for the maximum number of levels to crawl
        self.max_levels = int(input("Enter the maximum number of levels to crawl: "))
        
        # Mark the starting URL as processed only after Level 0 is complete
        # NOT before the crawl starts - this was causing the bug
        
        # ----- LEVEL 0: Initial Crawl -----
        self.console.print("\n[bold magenta]====== LEVEL 0 (Starting URL) ======[/bold magenta]")
        level_0_links_file = await self.crawl_level([self.start_url], 0)
        
        # NOW add the starting URL to processed links after it's been crawled
        self.processed_links.add(self.start_url)
        
        # ----- EVALUATION: Review Level 0 Links -----
        self.console.print("\n[bold magenta]====== EVALUATING LEVEL 0 LINKS ======[/bold magenta]")
        self.console.print("[bold yellow]STOPPING to evaluate links before proceeding to Level 1[/bold yellow]")
        clean_links = self.evaluate_links(level_0_links_file, 0)
        
        # ----- CONFIRMATION: Proceed to Level 1? -----
        if not Confirm.ask(f"\nProceed to Level 1 with {len(clean_links)} clean links?"):
            self.console.print("[bold yellow]Crawler stopped after Level 0.[/bold yellow]")
            return
        
        # Track the completed level (start with Level 0 completed)
        completed_levels = 0
        
        # ----- FIXED: Loop through ALL levels (1 to max_levels INCLUSIVE) -----
        for level in range(1, self.max_levels + 1):  # +1 ensures we include max_levels
            self.console.print(f"\n[bold magenta]====== LEVEL {level} ======[/bold magenta]")
            
            if not clean_links:
                self.console.print("[bold yellow]No clean links to crawl for this level. Stopping.[/bold yellow]")
                break
            
            # Crawl the next level with clean links
            level_links_file = await self.crawl_level(clean_links, level)
            
            # Update the completed level counter
            completed_levels = level
            
            # Evaluate links, even for the final level
            if level < self.max_levels:
                # Not the final level, ask to continue to next level
                self.console.print(f"\n[bold magenta]====== EVALUATING LEVEL {level} LINKS ======[/bold magenta]")
                self.console.print(f"[bold yellow]STOPPING to evaluate links before proceeding to Level {level+1}[/bold yellow]")
                clean_links = self.evaluate_links(level_links_file, level)
                
                # Confirm continuation to next level
                if not Confirm.ask(f"\nProceed to Level {level+1} with {len(clean_links)} clean links?"):
                    self.console.print(f"[bold yellow]Crawler stopped after Level {level}.[/bold yellow]")
                    break
            else:
                # This is the final level, still evaluate but don't ask about continuing
                self.console.print(f"\n[bold magenta]====== EVALUATING FINAL LEVEL {level} LINKS ======[/bold magenta]")
                self.console.print("[bold yellow]Performing final link evaluation[/bold yellow]")
                clean_links = self.evaluate_links(level_links_file, level)
                self.console.print(f"[bold green]Final clean list contains {len(clean_links)} links[/bold green]")
                self.console.print("[bold green]Reached maximum crawl depth. Would you like to crawl deeper?[/bold green]")
        
        # ----- ADDITIONAL LEVELS: Allow crawling beyond max_levels if desired -----
        while True:
            # Ask if the user wants to crawl deeper
            if not Confirm.ask(f"\nWould you like to crawl one more level (level {completed_levels + 1})?"):
                break
                
            # Set up next level
            next_level = completed_levels + 1
            self.console.print(f"\n[bold magenta]====== ADDITIONAL LEVEL {next_level} ======[/bold magenta]")
            
            if not clean_links:
                self.console.print("[bold yellow]No clean links to crawl for this level. Stopping.[/bold yellow]")
                break
            
            # Crawl the next level with clean links
            level_links_file = await self.crawl_level(clean_links, next_level)
            
            # Evaluate links
            self.console.print(f"\n[bold magenta]====== EVALUATING ADDITIONAL LEVEL {next_level} LINKS ======[/bold magenta]")
            clean_links = self.evaluate_links(level_links_file, next_level)
            
            # Update the completed level count
            completed_levels = next_level
        
        # Save global tracking information
        tracking_info = {
            "clean_links": list(self.all_clean_links),
            "rejected_links": list(self.rejected_links),
            "processed_links": list(self.processed_links),
            "visited_urls": list(self.global_visited_urls),
            "normalized_urls": len(self.global_normalized_urls),
            "normalized_base_urls": len(self.global_normalized_base_urls),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_clean_links": len(self.all_clean_links),
            "total_rejected_links": len(self.rejected_links),
            "total_processed_links": len(self.processed_links),
            "total_visited_urls": len(self.global_visited_urls)
        }
        
        with open(os.path.join(self.output_folder, "global_tracking.json"), "w", encoding="utf-8") as f:
            json.dump(tracking_info, f, indent=2)
        
        # Crawl complete
        self.console.print(f"\n[bold green]Crawl workflow completed with {completed_levels} levels![/bold green]")
        self.console.print(f"Results saved in: {self.output_folder}")
        self.console.print(f"Total unique clean links: {len(self.all_clean_links)}")
        self.console.print(f"Total unique rejected links: {len(self.rejected_links)}")
        self.console.print(f"Total unique processed links: {len(self.processed_links)}")
        self.console.print(f"Total unique visited URLs: {len(self.global_visited_urls)}")

async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Two-Phase Web Crawler with Gemini Evaluation")
    parser.add_argument("--url", required=True, help="Starting URL to crawl")
    parser.add_argument("--output", default="crawl_output", help="Base output folder")
    parser.add_argument("--api-key", help="Google Gemini API key (optional)")
    parser.add_argument("--min-delay", type=float, default=1.5, help="Minimum delay between requests")
    parser.add_argument("--max-delay", type=float, default=3.0, help="Maximum delay between requests")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Check for API key in environment variable if not provided as argument
    gemini_api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    
    # Initialize the coordinator
    coordinator = CrawlerCoordinator(
        start_url=args.url,
        output_folder=args.output,
        gemini_api_key=gemini_api_key,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        request_timeout=args.timeout
    )
    
    # Run the workflow
    await coordinator.run_workflow()

if __name__ == "__main__":
    asyncio.run(main())