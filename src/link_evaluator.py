import argparse
import json
import os
import time
import re
import random
import google.generativeai as genai
from urllib.parse import urlparse
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm, Prompt
from rich import print as rprint

class LinkEvaluator:
    def __init__(self, links_file, gemini_api_key, output_file="clean_links.json", 
                 decision_cache_file="gemini_decisions_cache.json",
                 override_patterns_file="override_patterns.json"):
        """
        Initialize the link evaluator with caching and deduplication features.
        
        Args:
            links_file (str): Path to the JSON file with links from the scraper
            gemini_api_key (str): API key for Google's Gemini API
            output_file (str): Path to save the cleaned links JSON
            decision_cache_file (str): Path to store cached Gemini decisions
            override_patterns_file (str): Path to store learned override patterns
        """
        self.links_file = links_file
        self.output_file = output_file
        self.console = Console()
        self.decision_cache_file = decision_cache_file
        self.override_patterns_file = override_patterns_file
        
        # Global sets for tracking URLs - prevent duplicates across pages
        self.all_keep_urls = set()
        self.all_reject_urls = set()
        self.pending_urls = set()  # URLs that still need evaluation
        
        # Load previous decisions cache if it exists
        self.decision_cache = self.load_decision_cache()
        
        # Load override patterns if they exist
        self.override_patterns = self.load_override_patterns()
        
        # Initialize Gemini API with better error handling
        genai.configure(api_key=gemini_api_key)
        self.initialize_gemini_model()
        
        # Known bad link patterns - MORE CONSERVATIVE now
        self.bad_link_patterns = [
            r"^mailto:",                # mailto links
            r"^tel:",                   # telephone links
            r"^javascript:",            # javascript links
            r"^#",                      # anchor links
            r"\.css$",                  # CSS files
            r"\.js$",                   # JavaScript files
            r"\.ico$",                  # Icon files
        ]
        
        # File extensions for direct downloads - used for conservative pre-filtering
        self.file_extensions = [
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
            '.zip', '.csv', '.jpg', '.jpeg', '.png', '.gif'
        ]

    def start_evaluation(self):
        """Start the evaluation process with options to reset learning."""
        self.console.print("[bold]Link Evaluation System[/bold]")
        
        # Offer to reset learned patterns
        if self.override_patterns and Confirm.ask("Would you like to reset all previously learned patterns for this session?"):
            self.override_patterns = {}
            self.save_override_patterns()
            self.console.print("[green]All learned patterns have been reset for this session.[/green]")
        
        # Offer to manage existing patterns
        if self.override_patterns and Confirm.ask("Would you like to view and manage existing patterns?"):
            self.manage_override_patterns()
        
        # Continue with normal evaluation
        self.process_all_pages()
    
    def load_override_patterns(self):
        """Load learned override patterns if available."""
        try:
            if os.path.exists(self.override_patterns_file):
                with open(self.override_patterns_file, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)
                    self.console.print(f"[green]Loaded {len(patterns)} override patterns[/green]")
                    return patterns
        except Exception as e:
            self.console.print(f"[yellow]Error loading override patterns: {str(e)}[/yellow]")
        
        return {}  # Empty patterns if file doesn't exist or has errors
    
    def save_override_patterns(self):
        """Save learned override patterns to file."""
        try:
            with open(self.override_patterns_file, 'w', encoding='utf-8') as f:
                json.dump(self.override_patterns, f, indent=2)
            self.console.print(f"[green]Saved {len(self.override_patterns)} override patterns[/green]")
        except Exception as e:
            self.console.print(f"[bold red]Error saving override patterns: {str(e)}[/bold red]")
    
    def remember_manual_override(self, url, original_decision, new_decision):
        """
        Remember when a human overrides an AI decision with more precise patterns.
        Stores the exact path, not a generalized pattern.
        """
        self.console.print(f"[bold cyan]Learning from override: {url}[/bold cyan]")
        
        # Don't generalize numbers anymore - store the exact URL path
        self.override_patterns[url] = new_decision
        
        # Save to a file for future runs
        self.save_override_patterns()
        
        self.console.print(f"[green]Learned exact path pattern: {url} -> {new_decision}[/green]")
    
    def apply_override_patterns(self, url):
        """
        Check if a URL matches any learned override patterns with much stricter matching.
        Only applies exact path matches or single-level path extensions.
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path

        # Create path segments for more precise matching
        path_segments = path.strip('/').split('/')

        for pattern, decision in self.override_patterns.items():
            pattern_parsed = urlparse(pattern) if '://' in pattern else urlparse(f"https://{pattern}")
            pattern_domain = pattern_parsed.netloc
            pattern_path = pattern_parsed.path

            # Only apply patterns from the same domain
            if domain != pattern_domain:
                continue

            # Case 1: Exact path match (e.g., /services/social-care exactly matches /services/social-care)
            if path == pattern_path:
                return True, decision

            # Case 2: Parent path exact match (e.g., /services matches /services/social-care, but ONLY if it's a direct parent)
            pattern_segments = pattern_path.strip('/').split('/')
            if len(pattern_segments) > 0 and len(path_segments) > len(pattern_segments):
                # Check if pattern is a direct parent path (e.g., /a/b is parent of /a/b/c but not of /a/b/c/d)
                if path_segments[:len(pattern_segments)] == pattern_segments and len(path_segments) == len(pattern_segments) + 1:
                    return True, decision

        return False, None
    
    def is_likely_file_url(self, url):
        """
        Check if a URL is likely a direct file download link (not a hub page).
        Only identifies obvious file links that apply across all local authorities.
        """
        # Standard file extensions - common across all authorities
        if any(url.lower().endswith(ext) for ext in self.file_extensions):
            return True
            
        # Only use very generic patterns that definitely indicate a direct file
        # These should apply across all local authorities
        direct_file_indicators = [
            r"/file/\d+/[^/]+$",     # File with ID ending the URL
            r"/attachment/\d+/[^/]+$" # Attachment with ID ending the URL
        ]
        
        for pattern in direct_file_indicators:
            if re.search(pattern, url):
                return True
        
        # Apply any learned patterns - but with domain-specific matching
        matches_override, decision = self.apply_override_patterns(url)
        if matches_override and decision == "REJECT":
            return True
        
        # For ambiguous cases, defer to Gemini
        return False
    
    def load_decision_cache(self):
        """Load previously cached Gemini decisions if available."""
        try:
            if os.path.exists(self.decision_cache_file):
                with open(self.decision_cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    self.console.print(f"[green]Loaded {len(cache)} cached decisions[/green]")
                    
                    # Populate the global tracking sets from the cache
                    for url, decision in cache.items():
                        if decision["decision"] == "KEEP":
                            self.all_keep_urls.add(url)
                        else:
                            self.all_reject_urls.add(url)
                            
                    return cache
        except Exception as e:
            self.console.print(f"[yellow]Error loading decision cache: {str(e)}[/yellow]")
        
        return {}  # Empty cache if file doesn't exist or has errors
    
    def save_decision_cache(self):
        """Save current Gemini decisions to cache file."""
        try:
            with open(self.decision_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.decision_cache, f, indent=2)
            self.console.print(f"[green]Saved {len(self.decision_cache)} decisions to cache[/green]")
        except Exception as e:
            self.console.print(f"[bold red]Error saving decision cache: {str(e)}[/bold red]")
    
    def initialize_gemini_model(self):
        """Initialize the Gemini model with proper error handling."""
        try:
            # Try to list available models to identify which ones are supported
            available_models = genai.list_models()
            model_names = [model.name for model in available_models]
            self.console.print(f"[green]Available models: {', '.join(model_names)}[/green]")
            
            # Look for specific Gemini models that are actually available
            # First try to find a Gemini 2.0 Flash model
            gemini_models_to_try = [
                # Try these exact models from the available list
                "models/gemini-2.0-flash",
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro",
                "models/gemini-pro-vision"  # Fallback to an older model
            ]
            
            # Try each model that's in the available models list
            for model_name in gemini_models_to_try:
                if model_name in model_names:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        self.console.print(f"[green]Successfully initialized {model_name}[/green]")
                        return
                    except Exception as e:
                        self.console.print(f"[yellow]Error initializing {model_name}: {str(e)}[/yellow]")
            
            # If none of our preferred models worked, try the first Gemini model in the list
            for model in model_names:
                if "gemini" in model.lower() and "embedding" not in model.lower():
                    try:
                        self.model = genai.GenerativeModel(model)
                        self.console.print(f"[green]Successfully initialized {model}[/green]")
                        return
                    except Exception as e:
                        self.console.print(f"[yellow]Error initializing {model}: {str(e)}[/yellow]")
            
            self.console.print("[yellow]No suitable Gemini models found. Will use rule-based filtering.[/yellow]")
            
        except Exception as e:
            self.console.print(f"[bold red]Error listing or initializing models: {str(e)}[/bold red]")
        
        # If we get here, no model could be initialized
        self.console.print("[yellow]Will continue with rule-based filtering only[/yellow]")
        self.model = None
    
    def manage_override_patterns(self):
        """Allow viewing and deleting specific patterns."""
        self.console.print("\n[bold]Learned URL Patterns:[/bold]")
        
        if not self.override_patterns:
            self.console.print("[yellow]No patterns have been learned yet.[/yellow]")
            return
        
        # Display patterns in a table
        table = Table(title="Learned Patterns")
        table.add_column("#", style="dim")
        table.add_column("Pattern", style="cyan")
        table.add_column("Decision", style="green")
        
        patterns = list(self.override_patterns.items())
        for idx, (pattern, decision) in enumerate(patterns):
            decision_style = "green" if decision == "KEEP" else "red"
            table.add_row(str(idx+1), pattern, f"[{decision_style}]{decision}[/{decision_style}]")
        
        self.console.print(table)
        
        # Options for management
        self.console.print("\n[bold]Options:[/bold]")
        self.console.print("1. Delete specific pattern")
        self.console.print("2. Delete all patterns")
        self.console.print("3. Return to main menu")
        
        choice = Prompt.ask("Enter your choice", choices=["1", "2", "3"], default="3")
        
        if choice == "1":
            pattern_idx = Prompt.ask("Enter pattern number to delete", default="1")
            try:
                idx = int(pattern_idx) - 1
                if 0 <= idx < len(patterns):
                    pattern = patterns[idx][0]
                    del self.override_patterns[pattern]
                    self.save_override_patterns()
                    self.console.print(f"[green]Deleted pattern: {pattern}[/green]")
                else:
                    self.console.print("[red]Invalid pattern number[/red]")
            except ValueError:
                self.console.print("[red]Invalid input[/red]")
        
        elif choice == "2":
            if Confirm.ask("Are you sure you want to delete ALL learned patterns?"):
                self.override_patterns = {}
                self.save_override_patterns()
                self.console.print("[green]All patterns deleted[/green]")
        
    def analyze_links_with_rules(self, page_index):
        """Use rule-based approach when Gemini API is not available."""
        page = self.pages[page_index]
        page_url = page.get("url", "Unknown URL")
        prefiltered_links = page.get("prefiltered_links", [])
        
        self.console.print(f"[yellow]Using rule-based filtering for {len(prefiltered_links)} links[/yellow]")
        
        # Simple patterns to identify navigation vs content
        navigation_patterns = [
            r"/about", r"/contact", r"/accessibility", r"/privacy", r"/cookies",
            r"/terms", r"/sitemap", r"/feedback", r"/help", r"/faq", r"/support",
            r"^/$", r"/home", r"/index", r"/a-z", r"/search"
        ]
        
        content_patterns = [
            r"/services/", r"/guide/", r"/article/", r"/information/", 
            r"/forms/", r"/applications/", r"/benefits/", r"/health/",
            r"/social-care/", r"/children/", r"/adults/", r"/housing/",
            r"/education/", r"/schools/", r"/leisure/", r"/environment/"
        ]
        
        analysis_results = []
        
        for link in prefiltered_links:
            # Check if we already have a cached decision for this URL
            if link in self.decision_cache:
                cached_decision = self.decision_cache[link]
                analysis_results.append({
                    "url": link,
                    "decision": cached_decision["decision"],
                    "reason": cached_decision["reason"] + " (cached)",
                    "confidence": cached_decision.get("confidence", 1.0)
                })
                continue
            
            # First check override patterns from previous human corrections
            matches_override, override_decision = self.apply_override_patterns(link)
            if matches_override:
                decision = override_decision
                reason = "Applied learned pattern from human override"
                confidence = 1.0  # High confidence for learned patterns
            else:    
                # Check if it's a file URL
                if self.is_likely_file_url(link):
                    decision = "REJECT"
                    reason = "Direct file download URL - will be downloaded but not crawled"
                    confidence = 0.9  # High confidence for file detection
                else:
                    # Default to keeping links
                    decision = "KEEP"
                    reason = "Default content page"
                    confidence = 0.6  # Moderate confidence for default
                    
                    # Check for navigation patterns
                    for pattern in navigation_patterns:
                        if re.search(pattern, link, re.IGNORECASE):
                            decision = "REJECT"
                            reason = f"Navigation link (matches {pattern})"
                            confidence = 0.7  # Good confidence for navigation patterns
                            break
                    
                    # Content patterns override navigation patterns
                    for pattern in content_patterns:
                        if re.search(pattern, link, re.IGNORECASE):
                            decision = "KEEP"
                            reason = f"Content page (matches {pattern})"
                            confidence = 0.8  # Higher confidence for content patterns
                            break
            
            # Add the result
            analysis_results.append({
                "url": link,
                "decision": decision,
                "reason": reason,
                "confidence": confidence
            })
            
            # Cache this decision
            self.decision_cache[link] = {
                "decision": decision,
                "reason": reason,
                "source": "rule-based",
                "confidence": confidence
            }
        
        return analysis_results
        
    def load_links(self):
        """Load links from the JSON file produced by the scraper."""
        self.console.print(f"Loading links from [bold]{self.links_file}[/bold]")
        try:
            with open(self.links_file, 'r', encoding='utf-8') as f:
                links_data = json.load(f)
            
            # Check the structure of the loaded data
            if isinstance(links_data, list):
                # This is the all_collected_links format
                self.pages = links_data
            else:
                # This is a single page links format
                self.pages = [links_data]
                
            self.console.print(f"Loaded data for [bold]{len(self.pages)}[/bold] pages")
            return True
        except Exception as e:
            self.console.print(f"[bold red]Error loading links file:[/bold red] {str(e)}")
            return False

    def prefilter_links(self):
        """Apply minimal pre-filtering based on obvious patterns before using Gemini."""
        total_links = 0
        prefilterd_links = 0
        
        for page in self.pages:
            # Original internal links
            internal_links = page.get("internal_links", [])
            total_links += len(internal_links)
            
            # Apply pre-filtering
            filtered_links = []
            rejected_links = []
            
            for link in internal_links:
                # Skip links that have already been processed in previous pages
                if link in self.all_keep_urls or link in self.all_reject_urls:
                    continue
                    
                should_reject = False
                rejection_reason = None
                
                # CONSERVATIVE PRE-FILTERING: Only reject obvious non-content URLs
                # 1. Only reject links with obvious file extensions (e.g., .pdf, .doc)
                if any(link.lower().endswith(ext) for ext in self.file_extensions):
                    should_reject = True
                    rejection_reason = "Direct file link with extension - will be downloaded but not crawled"
                # 2. Reject only the most obvious non-content links
                else:
                    for pattern in self.bad_link_patterns:
                        if re.search(pattern, link):
                            should_reject = True
                            rejection_reason = f"Matches exclude pattern: {pattern}"
                            break
                
                if should_reject:
                    rejected_links.append({"url": link, "reason": rejection_reason})
                    self.all_reject_urls.add(link)  # Add to global rejection set
                    
                    # Add to decision cache
                    self.decision_cache[link] = {
                        "decision": "REJECT",
                        "reason": rejection_reason,
                        "source": "pattern-match",
                        "confidence": 0.95  # High confidence for explicit patterns
                    }
                else:
                    filtered_links.append(link)
                    self.pending_urls.add(link)  # Mark as pending evaluation
                
            # Store the filtered links in the page data
            page["prefiltered_links"] = filtered_links
            page["rejected_links"] = rejected_links
            prefilterd_links += len(rejected_links)
        
        self.console.print(f"Pre-filtered [bold]{prefilterd_links}[/bold] out of [bold]{total_links}[/bold] links")
        self.console.print(f"Removed [bold]{total_links - prefilterd_links - len(self.pending_urls)}[/bold] duplicate links")
        return prefilterd_links

    def create_url_display(self, url, max_length=60):
        """Create a more readable URL display for tables."""
        parsed = urlparse(url)
        
        # Show only domain + path for cleaner display
        domain = parsed.netloc
        path = parsed.path
        
        # If URL is too long, truncate the middle part
        if len(url) > max_length:
            # Show domain and important parts of the path
            path_parts = path.split('/')
            if len(path_parts) > 3:
                # Keep first and last two path segments
                shortened_path = '/'.join([path_parts[0], '...', path_parts[-2], path_parts[-1]])
            else:
                shortened_path = path
            
            display_url = f"{domain}{shortened_path}"
            if len(display_url) > max_length:
                display_url = display_url[:max_length-3] + "..."
            
            # Add a note that this is shortened
            return f"{display_url} [dim](shortened)[/dim]"
        else:
            return url

    def view_full_urls(self, urls, title="URLs"):
        """Display full URLs with ability to scroll and view details."""
        self.console.print(f"\n[bold]{title}[/bold]")
        
        for i, url in enumerate(urls):
            self.console.print(f"\n[bold]{i+1}.[/bold] [cyan]{url}[/cyan]")
            
            # Display parsed components for easier understanding
            parsed = urlparse(url)
            self.console.print(f"  Domain: {parsed.netloc}")
            self.console.print(f"  Path: {parsed.path}")
            if parsed.query:
                self.console.print(f"  Query: {parsed.query}")
            if parsed.fragment:
                self.console.print(f"  Fragment: {parsed.fragment}")
            
            # For long lists, offer to continue or stop
            if i < len(urls) - 1 and (i + 1) % 5 == 0:
                if not Confirm.ask("Show more URLs?"):
                    break

    def calculate_confidence_score(self, link, gemini_response):
        """
        Calculate a confidence score for Gemini's decision.
        
        Args:
            link (str): The URL being analyzed
            gemini_response (dict): The response from Gemini
        
        Returns:
            float: A confidence score between 0.0 and 1.0
        """
        # Default medium confidence
        confidence = 0.5
        
        # 1. If it matches a pattern we've learned from human overrides, high confidence
        matches_override, _ = self.apply_override_patterns(link)
        if matches_override:
            return 1.0
            
        # 2. If the URL obviously contains file indicators, higher confidence
        if any(ext in link.lower() for ext in self.file_extensions):
            confidence += 0.3
            
        # 3. Look for content vs. navigation indicators in the URL
        content_indicators = ['service', 'guide', 'information', 'article', 'content', 'page']
        navigation_indicators = ['about', 'contact', 'login', 'search', 'home', 'index']
        
        # Check for content indicators
        for indicator in content_indicators:
            if indicator in link.lower():
                if gemini_response["decision"] == "KEEP":
                    confidence += 0.1  # Agreement with URL pattern
                else:
                    confidence -= 0.1  # Disagreement with URL pattern
                break
                
        # Check for navigation indicators
        for indicator in navigation_indicators:
            if indicator in link.lower():
                if gemini_response["decision"] == "REJECT":
                    confidence += 0.1  # Agreement with URL pattern
                else:
                    confidence -= 0.1  # Disagreement with URL pattern
                break
                
        # 4. Check if Gemini's reasoning is detailed or vague
        reason = gemini_response.get("reason", "").lower()
        if len(reason) > 30 and ("file" in reason or "content" in reason or "hub" in reason):
            confidence += 0.1  # More detailed reasoning
        
        # 5. Ambiguous links with 'download' but potentially hub pages
        if "download" in link.lower() and not any(link.lower().endswith(ext) for ext in self.file_extensions):
            confidence -= 0.2  # These are often ambiguous
            
        # Ensure confidence is between 0.0 and 1.0
        return max(0.0, min(1.0, confidence))

    def analyze_links_with_gemini(self, page_index):
        """Use Gemini to analyze links from a specific page in batches of 30."""
        page = self.pages[page_index]
        page_url = page.get("url", "Unknown URL")
        prefiltered_links = page.get("prefiltered_links", [])
        
        if not prefiltered_links:
            self.console.print(f"[yellow]No links to analyze for page {page_index + 1}: {page_url}[/yellow]")
            return []
        
        self.console.print(f"\n[bold]Analyzing {len(prefiltered_links)} links for page {page_index + 1}: {page_url}[/bold]")
        
        # Check if Gemini model is available
        if self.model is None:
            self.console.print("[yellow]Gemini API not available. Using rule-based filtering.[/yellow]")
            return self.analyze_links_with_rules(page_index)
        
        # First, check for learned override patterns and cached decisions to reduce API calls
        uncached_links = []
        precalculated_results = []
        
        for link in prefiltered_links:
            # First check for learned override patterns
            matches_override, override_decision = self.apply_override_patterns(link)
            if matches_override:
                precalculated_results.append({
                    "url": link,
                    "decision": override_decision,
                    "reason": "Applied learned pattern from human override",
                    "confidence": 1.0  # High confidence for human-verified patterns
                })
                
                # Update global tracking sets
                if override_decision == "KEEP":
                    self.all_keep_urls.add(link)
                    if link in self.pending_urls:
                        self.pending_urls.remove(link)
                else:
                    self.all_reject_urls.add(link)
                    if link in self.pending_urls:
                        self.pending_urls.remove(link)
                        
                # Add to decision cache
                self.decision_cache[link] = {
                    "decision": override_decision,
                    "reason": "Applied learned pattern from human override",
                    "source": "override-pattern",
                    "confidence": 1.0
                }
                continue
                
            # Then check cached decisions
            if link in self.decision_cache:
                cached_decision = self.decision_cache[link]
                precalculated_results.append({
                    "url": link,
                    "decision": cached_decision["decision"],
                    "reason": cached_decision["reason"] + " (cached)",
                    "confidence": cached_decision.get("confidence", 0.5)
                })
                
                # Update global tracking sets based on cached decision
                if cached_decision["decision"] == "KEEP":
                    self.all_keep_urls.add(link)
                    if link in self.pending_urls:
                        self.pending_urls.remove(link)
                else:
                    self.all_reject_urls.add(link)
                    if link in self.pending_urls:
                        self.pending_urls.remove(link)
            else:
                uncached_links.append(link)
        
        if precalculated_results:
            self.console.print(f"[green]Using {len(precalculated_results)} pre-calculated decisions, {len(uncached_links)} links need Gemini evaluation[/green]")
        
        # If all links were pre-calculated, return those results
        if not uncached_links:
            return precalculated_results
        
        # Process remaining links in batches of 30
        batch_size = 30
        all_analysis_results = precalculated_results.copy()  # Start with pre-calculated results
        total_batches = (len(uncached_links) + batch_size - 1) // batch_size  # Ceiling division
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(uncached_links))
            batch_links = uncached_links[start_idx:end_idx]
            
            self.console.print(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_links)} links)")
            
            # Create a prompt for Gemini that explains what we're looking for
            prompt = f"""
            I'm scraping a local authority social care website and need to filter out irrelevant internal links.

            The current page is: {page_url}

            I have a list of internal links from this page, and I need your help to identify which ones should be kept for crawling and which should be rejected.

            IMPORTANT DISTINCTION FOR LOCAL AUTHORITY WEBSITES:

            - REJECT links that clearly point to single downloadable files:
            * Links that directly serve a single downloadable file (PDF, DOC, etc.)
            * URLs that end with file extensions like .pdf, .doc, .xlsx
            * They represent an endpoint in the crawl (we'll download them but not crawl them further)
            * These are typically individual documents with no browsing interface

            - KEEP hub/collection pages that organize multiple documents:
            * Pages that list or categorize multiple downloadable resources
            * Pages with URLs that contain words like 'downloads', 'documents', or 'publications' followed by a category name
            * They serve as navigation pages that users would browse to find documents
            * They are important junction points that lead to multiple resources
            * IMPORTANT: Even if these pages have 'download' in the URL, they should be KEPT if they're collection pages

            - KEEP normal content pages with information:
            * Regular website pages with text content, not just file downloads
            * Pages about social care services, information, budget, or resources
            * Pages that users would browse to learn about topics

            - REJECT navigation and utility links:
            * Header/footer navigation links
            * Utility functions like print, share, login
            * Duplicate links with different parameters

            IMPORTANT: A URL may contain words like 'download' but still be a collection/hub page rather than a direct file download. Consider the context and purpose of the page, not just the URL keywords.

            When in doubt about a URL's purpose, prefer KEEP to ensure we don't miss valuable content.

            Examples of what to REJECT:
            - Links that are clearly direct downloads of individual PDFs, documents, or media files
            - URLs that end with file extensions like .pdf, .docx, .xlsx
            - Navigation links, login pages, utility functions

            Examples of what to KEEP:
            - Pages that likely contain multiple file downloads (document libraries)
            - Content pages with information about social care
            - Hub pages that organize documents into categories, even if they have 'download' in the URL

            Here are the links to evaluate (batch {batch_num + 1} of {total_batches}):
            {batch_links}

            Format your response as a JSON array like this:
            [
            {{"url": "link1", "decision": "KEEP", "reason": "Content page about services"}},
            {{"url": "link2", "decision": "REJECT", "reason": "Navigation menu link"}},
            {{"url": "link3", "decision": "REJECT", "reason": "Direct file download link"}},
            {{"url": "link4", "decision": "KEEP", "reason": "Hub page with multiple file downloads"}}
            ]

            Be very decisive and consistent in your evaluations. The same type of link should always get the same decision.
            Only output valid JSON without any extra text.
            """
            
            # Try multiple times with increasing delays for rate limiting or temporary issues
            max_retries = 3
            batch_results = None
            
            for retry in range(max_retries):
                try:
                    # Send the prompt to Gemini and get the response
                    response = self.model.generate_content(prompt)
                    
                    # Extract the JSON from the response
                    try:
                        # First try to parse the response text directly as JSON
                        batch_results = json.loads(response.text)
                        break  # Success, exit retry loop
                    except json.JSONDecodeError:
                        # If that fails, try to extract JSON using regex
                        match = re.search(r'\[\s*\{.*\}\s*\]', response.text, re.DOTALL)
                        if match:
                            batch_results = json.loads(match.group(0))
                            break  # Success, exit retry loop
                        else:
                            raise ValueError(f"Could not extract JSON from Gemini response for batch {batch_num + 1}")
                
                except Exception as e:
                    self.console.print(f"[yellow]Error on attempt {retry+1}: {str(e)}[/yellow]")
                    if retry < max_retries - 1:
                        delay = (retry + 1) * 3  # Increasing delay: 3s, 6s, 9s
                        self.console.print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        self.console.print(f"[bold red]Failed after {max_retries} attempts for batch {batch_num + 1}[/bold red]")
                        # Fall back to rule-based analysis for this batch
                        batch_results = []
                        for link in batch_links:
                            # Simple heuristic for decision
                            if self.is_likely_file_url(link):
                                decision = "REJECT"
                                reason = "Direct file download URL (fallback)"
                                confidence = 0.7
                            elif any(pattern in link.lower() for pattern in ['/about', '/contact', '/terms', '/privacy']):
                                decision = "REJECT"
                                reason = "Likely navigation/utility page (fallback)"
                                confidence = 0.6
                            else:
                                decision = "KEEP"
                                reason = "Potential content page (fallback)"
                                confidence = 0.4  # Lower confidence for fallback
                                
                            batch_results.append({
                                "url": link,
                                "decision": decision,
                                "reason": reason
                            })
            
            if batch_results:
                # Process each result, adding confidence scores
                for item in batch_results:
                    url = item.get("url")
                    if url and "decision" in item and "reason" in item:
                        # Calculate confidence score for this decision
                        confidence = self.calculate_confidence_score(url, item)
                        item["confidence"] = confidence
                        
                        # Cache the decision with confidence
                        self.decision_cache[url] = {
                            "decision": item["decision"],
                            "reason": item["reason"],
                            "source": "gemini",
                            "confidence": confidence
                        }
                        
                        # Update global tracking sets
                        if item["decision"] == "KEEP":
                            self.all_keep_urls.add(url)
                            if url in self.pending_urls:
                                self.pending_urls.remove(url)
                        else:
                            self.all_reject_urls.add(url)
                            if url in self.pending_urls:
                                self.pending_urls.remove(url)
                
                # Add batch results to the combined results
                all_analysis_results.extend(batch_results)
                
                # Add a delay between batches to avoid rate limiting
                if batch_num < total_batches - 1:
                    delay_time = 3  # 3-second delay between batches
                    self.console.print(f"Waiting {delay_time} seconds before processing next batch...")
                    time.sleep(delay_time)
            
        # Update the page data with all Gemini's analysis results
        page["gemini_analysis"] = all_analysis_results
        
        # Flag any low-confidence items
        low_confidence_items = [item for item in all_analysis_results if item.get("confidence", 0) < 0.6]
        if low_confidence_items:
            self.console.print(f"[yellow]Found {len(low_confidence_items)} low-confidence decisions that may need manual review[/yellow]")
        
        self.console.print(f"[green]Successfully analyzed {len(all_analysis_results)} out of {len(prefiltered_links)} links[/green]")
        
        # Save the updated cache to file
        self.save_decision_cache()
        
        # Return the combined analysis results
        return all_analysis_results

    def process_all_pages(self):
        """Process all pages and generate clean link lists."""
        if not hasattr(self, 'pages'):
            if not self.load_links():
                return False
        
        # Pre-filter links based on known patterns
        self.prefilter_links()
        
        # Process each page with Gemini
        for i in range(len(self.pages)):
            # Analyze links with Gemini
            analysis = self.analyze_links_with_gemini(i)
            
            if analysis:
                # Apply Gemini's recommendations
                keep_links = []
                reject_links = []
                low_confidence_links = []
                
                for item in analysis:
                    # Check if the item has the required fields
                    if "url" not in item or "decision" not in item or "reason" not in item:
                        self.console.print(f"[yellow]Warning: Skipping malformed analysis item: {item}[/yellow]")
                        continue
                        
                    # Flag low confidence items for manual review
                    confidence = item.get("confidence", 0.5)
                    is_low_confidence = confidence < 0.6
                    
                    if is_low_confidence:
                        # Add to low confidence list for special handling
                        low_confidence_links.append({
                            "url": item["url"], 
                            "decision": item["decision"], 
                            "reason": item["reason"],
                            "confidence": confidence
                        })
                    
                    if item["decision"].upper() == "KEEP":
                        keep_links.append({
                            "url": item["url"], 
                            "reason": item["reason"],
                            "confidence": confidence,
                            "low_confidence": is_low_confidence
                        })
                    elif item["decision"].upper() == "REJECT":
                        reject_links.append({
                            "url": item["url"], 
                            "reason": item["reason"],
                            "confidence": confidence,
                            "low_confidence": is_low_confidence
                        })
                    else:
                        self.console.print(f"[yellow]Warning: Unknown decision '{item['decision']}' for {item['url']}[/yellow]")
                
                # Update the page data
                self.pages[i]["gemini_keep_links"] = keep_links
                self.pages[i]["gemini_reject_links"] = reject_links
                self.pages[i]["low_confidence_links"] = low_confidence_links
                
                # Summary for this page
                self.console.print(f"Page {i+1}: Keeping {len(keep_links)} links, rejecting {len(reject_links)} links, {len(low_confidence_links)} low-confidence decisions")
                
                # Delay to avoid rate limiting with Gemini API
                if i < len(self.pages) - 1:  # Don't delay after the last page
                    time.sleep(2)
            else:
                # If no analysis results, use all prefiltered links for manual review
                self.console.print(f"[yellow]No analysis results for page {i+1} - using all prefiltered links for manual review[/yellow]")
                prefiltered_links = self.pages[i].get("prefiltered_links", [])
                keep_links = [{"url": link, "reason": "No AI analysis - needs manual review", "confidence": 0.0, "low_confidence": True} for link in prefiltered_links]
                self.pages[i]["gemini_keep_links"] = keep_links
                self.pages[i]["gemini_reject_links"] = []
                self.pages[i]["low_confidence_links"] = keep_links.copy()
        
        # Generate the final clean links
        self.generate_clean_links()
        return True

    def generate_clean_links(self):
        """Generate a clean list of links based on Gemini's analysis."""
        clean_links = []
        
        for page in self.pages:
            # Get the links that Gemini recommended to keep
            keep_links = [item["url"] for item in page.get("gemini_keep_links", [])]
            
            # Add to the clean links list
            for link in keep_links:
                if link not in clean_links:
                    clean_links.append(link)
        
        # Save the clean links to a file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "clean_links": clean_links,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_count": len(clean_links)
            }, f, indent=2)
        
        self.console.print(f"\n[bold green]Generated {len(clean_links)} clean links, saved to {self.output_file}[/bold green]")
        
        # Return the clean links
        return clean_links

    def display_gemini_analysis(self):
        """Display Gemini's analysis for manual review, one page at a time."""
        if not hasattr(self, 'pages'):
            if not self.load_links():
                return
        
        total_kept = 0
        total_rejected = 0
        total_overrides = 0
        
        # New page-by-page review interface
        self.console.print("\n[bold cyan]Page-by-Page Review Interface[/bold cyan]")
        self.console.print("Review each page's links individually before proceeding to the next page.\n")
        
        # Process each page separately
        for i, page in enumerate(self.pages):
            page_url = page.get("url", "Unknown URL")
            keep_links = page.get("gemini_keep_links", [])
            reject_links = page.get("gemini_reject_links", [])
            low_confidence_links = page.get("low_confidence_links", [])
            
            if not keep_links and not reject_links:
                self.console.print(f"\n[yellow]Page {i+1}: {page_url} - No analysis results[/yellow]")
                continue
            
            # Display a clear page header with a divider line
            self.console.print("\n" + "=" * 80)
            self.console.print(f"[bold blue]PAGE {i+1}: {page_url}[/bold blue]")
            self.console.print("=" * 80)
            
            # Display low confidence links that need review
            if low_confidence_links:
                low_conf_table = Table(title=f"Low Confidence Decisions - REVIEW THESE FIRST ({len(low_confidence_links)})")
                low_conf_table.add_column("#", style="dim", width=4)
                low_conf_table.add_column("URL", style="yellow")
                low_conf_table.add_column("Decision", width=10)
                low_conf_table.add_column("Confidence", width=10)
                low_conf_table.add_column("Reason")
                
                for idx, item in enumerate(low_confidence_links):
                    decision_style = "green" if item["decision"] == "KEEP" else "red"
                    confidence = item.get("confidence", 0)
                    low_conf_table.add_row(
                        str(idx+1), 
                        self.create_url_display(item["url"]), 
                        f"[{decision_style}]{item['decision']}[/{decision_style}]",
                        f"{confidence:.2f}",
                        item["reason"]
                    )
                
                self.console.print(low_conf_table)
                self.console.print("[yellow bold]These links have low confidence scores and should be manually verified.[/yellow bold]")
                
                # Option to view full URLs before reviewing
                if Confirm.ask("View full URLs before reviewing?"):
                    self.view_full_urls([item["url"] for item in low_confidence_links], "Low Confidence URLs")
                
                # Ask if user wants to review these low confidence links
                if Confirm.ask("Review these low confidence links now?"):
                    self.review_low_confidence_links(low_confidence_links, page_url, page)
                    
                    # ADD THESE LINES to refresh the link lists after review
                    keep_links = page.get("gemini_keep_links", [])  
                    reject_links = page.get("gemini_reject_links", [])
            
            # Display links to keep
            if keep_links:
                keep_table = Table(title=f"Links to Keep ({len(keep_links)})")
                keep_table.add_column("#", style="dim", width=4)
                keep_table.add_column("URL", style="green")
                keep_table.add_column("Confidence", width=10)
                keep_table.add_column("Reason")
                
                # Display links that aren't low confidence
                high_conf_keeps = [item for item in keep_links if not item.get("low_confidence", False)]
                
                for idx, item in enumerate(high_conf_keeps):
                    confidence = item.get("confidence", 0.5)
                    keep_table.add_row(
                        str(idx+1), 
                        self.create_url_display(item["url"]), 
                        f"{confidence:.2f}", 
                        item["reason"]
                    )
                
                self.console.print(keep_table)
                total_kept += len(keep_links)
                
                # Option to view full URLs
                if Confirm.ask("View full URLs for links to keep?"):
                    self.view_full_urls([item["url"] for item in high_conf_keeps], "Links to Keep (Full URLs)")
            
            # Display links to reject
            if reject_links:
                reject_table = Table(title=f"Links to Reject ({len(reject_links)})")
                reject_table.add_column("#", style="dim", width=4)
                reject_table.add_column("URL", style="red")
                reject_table.add_column("Confidence", width=10)
                reject_table.add_column("Reason")
                
                # Display links that aren't low confidence
                high_conf_rejects = [item for item in reject_links if not item.get("low_confidence", False)]
                
                for idx, item in enumerate(high_conf_rejects):
                    confidence = item.get("confidence", 0.5)
                    reject_table.add_row(
                        str(idx+1), 
                        self.create_url_display(item["url"]), 
                        f"{confidence:.2f}", 
                        item["reason"]
                    )
                
                self.console.print(reject_table)
                total_rejected += len(reject_links)
                
                # Option to view full URLs
                if Confirm.ask("View full URLs for links to reject?"):
                    self.view_full_urls([item["url"] for item in high_conf_rejects], "Links to Reject (Full URLs)")
            
            # Ask if user wants to modify this page's links
            if Confirm.ask("Do you want to modify this page's links?"):
                original_keep_links = [item.copy() for item in keep_links]
                modified_keep_links = self.modify_page_links(keep_links, reject_links, page_url)
                page["gemini_keep_links"] = modified_keep_links
                
                # Check for overrides to learn from
                for old_item in original_keep_links:
                    # If an item was in the keep links but is no longer there, it was moved to reject
                    if not any(new_item["url"] == old_item["url"] for new_item in modified_keep_links):
                        # This was changed from KEEP to REJECT
                        self.remember_manual_override(old_item["url"], "KEEP", "REJECT")
                        total_overrides += 1
                
                # Check if any links were moved from reject to keep
                original_reject_urls = {item["url"] for item in reject_links}
                for new_item in modified_keep_links:
                    if new_item["url"] in original_reject_urls:
                        # This was changed from REJECT to KEEP
                        self.remember_manual_override(new_item["url"], "REJECT", "KEEP")
                        total_overrides += 1
                
                # Update the global tracking sets
                for item in modified_keep_links:
                    url = item["url"]
                    self.all_keep_urls.add(url)
                    if url in self.all_reject_urls:
                        self.all_reject_urls.remove(url)
                    
                    # Update the cache with the user's decision
                    self.decision_cache[url] = {
                        "decision": "KEEP",
                        "reason": item["reason"] + " (manually reviewed)",
                        "source": "manual",
                        "confidence": 1.0  # High confidence for manual decisions
                    }
                
                # Show updated count
                self.console.print(f"[green]Updated: Now keeping {len(modified_keep_links)} links from this page[/green]")
                if total_overrides > 0:
                    self.console.print(f"[cyan]Learned from {total_overrides} manual overrides[/cyan]")
                
                # Save the updated cache
                self.save_decision_cache()
            
            # Ask to continue to next page
            if i < len(self.pages) - 1:  # Don't ask after the last page
                if not Confirm.ask("Continue to next page?"):
                    break
        
        # Display totals
        self.console.print("\n" + "=" * 80)
        self.console.print(f"[bold green]Review Complete: Keeping {total_kept} links, rejected {total_rejected} links[/bold green]")
        if total_overrides > 0:
            self.console.print(f"[bold cyan]Learned from {total_overrides} manual overrides - system will improve over time[/bold cyan]")
        
        # Re-generate clean links file with any modifications
        self.generate_clean_links()

    def review_low_confidence_links(self, low_confidence_links, page_url, page):
        """Specifically review low confidence links and learn from overrides."""
        self.console.print("\n[bold]Review Low Confidence Links for Page:[/bold] " + page_url)
        
        total_overrides = 0
        
        for idx, item in enumerate(low_confidence_links):
            url = item["url"]
            current_decision = item["decision"]
            confidence = item.get("confidence", 0)
            
            self.console.print(f"\n[bold]{idx+1}/{len(low_confidence_links)}[/bold]: {url}")
            self.console.print(f"Current decision: [{current_decision}] (confidence: {confidence:.2f})")
            self.console.print(f"Reason: {item['reason']}")
            
            # Ask for the correct decision
            options = ["k", "r", "s"] if current_decision == "KEEP" else ["k", "r", "s"]
            prompt = "Override? (k)eep, (r)eject, or (s)kip" 
            
            choice = Prompt.ask(prompt, choices=options, default="s")
            
            if choice == "s":
                # Skip this link
                continue
            elif choice == "k" and current_decision != "KEEP":
                # Override: Change from REJECT to KEEP
                new_decision = "KEEP"
                self.remember_manual_override(url, current_decision, new_decision)
                
                # Update the global tracking and decision cache
                self.all_keep_urls.add(url)
                if url in self.all_reject_urls:
                    self.all_reject_urls.remove(url)
                
                self.decision_cache[url] = {
                    "decision": new_decision,
                    "reason": "Manually overridden from REJECT to KEEP",
                    "source": "manual",
                    "confidence": 1.0
                }
                
                # Update the page's keep/reject lists
                keep_item = item.copy()
                keep_item["decision"] = new_decision
                keep_item["reason"] = "Manually overridden from REJECT to KEEP"
                keep_item["confidence"] = 1.0
                keep_item["low_confidence"] = False
                
                page["gemini_keep_links"].append(keep_item)
                page["gemini_reject_links"] = [r for r in page["gemini_reject_links"] if r["url"] != url]
                
                self.console.print(f"[green]Changed decision to KEEP[/green]")
                total_overrides += 1
                
            elif choice == "r" and current_decision != "REJECT":
                # Override: Change from KEEP to REJECT
                new_decision = "REJECT"
                self.remember_manual_override(url, current_decision, new_decision)
                
                # Update the global tracking and decision cache
                self.all_reject_urls.add(url)
                if url in self.all_keep_urls:
                    self.all_keep_urls.remove(url)
                
                self.decision_cache[url] = {
                    "decision": new_decision,
                    "reason": "Manually overridden from KEEP to REJECT",
                    "source": "manual",
                    "confidence": 1.0
                }
                
                # Update the page's keep/reject lists
                reject_item = item.copy()
                reject_item["decision"] = new_decision
                reject_item["reason"] = "Manually overridden from KEEP to REJECT"
                reject_item["confidence"] = 1.0
                reject_item["low_confidence"] = False
                
                page["gemini_reject_links"].append(reject_item)
                page["gemini_keep_links"] = [k for k in page["gemini_keep_links"] if k["url"] != url]
                
                self.console.print(f"[red]Changed decision to REJECT[/red]")
                total_overrides += 1
                
        # Update low confidence links to remove the ones we've reviewed
        page["low_confidence_links"] = [
            item for item in page["low_confidence_links"] 
            if self.decision_cache.get(item["url"], {}).get("source") != "manual"
        ]
        
        # Save changes
        self.save_decision_cache()
        self.save_override_patterns()
        
        self.console.print(f"[bold cyan]Made {total_overrides} decision overrides for low confidence links[/bold cyan]")

    def modify_page_links(self, keep_links, reject_links, page_url):
        """Allow the user to modify links for a specific page."""
        self.console.print("\n[bold]Modify Links for Page:[/bold] " + page_url)
        
        # Create a combined list of all links with their status
        all_links = []
        for idx, item in enumerate(keep_links):
            all_links.append({
                "idx": idx,
                "url": item["url"],
                "status": "KEEP",
                "reason": item["reason"],
                "confidence": item.get("confidence", 0.5)
            })
        
        for idx, item in enumerate(reject_links):
            all_links.append({
                "idx": idx + len(keep_links),
                "url": item["url"],
                "status": "REJECT",
                "reason": item["reason"],
                "confidence": item.get("confidence", 0.5)
            })
        
        # Display options
        self.console.print("\n[bold]Options:[/bold]")
        self.console.print("1. Move links from REJECT to KEEP")
        self.console.print("2. Move links from KEEP to REJECT")
        self.console.print("3. View full URLs in a list")
        self.console.print("4. Keep current selection")
        
        choice = Prompt.ask("Enter your choice", choices=["1", "2", "3", "4"], default="4")
        
        if choice == "1":
            # Show rejected links
            if reject_links:
                reject_table = Table(title="Links Currently Rejected")
                reject_table.add_column("#", style="dim", width=4)
                reject_table.add_column("URL", style="red")
                reject_table.add_column("Confidence", width=10)
                reject_table.add_column("Reason")
                
                for idx, item in enumerate(reject_links):
                    confidence = item.get("confidence", 0.5)
                    reject_table.add_row(str(idx+1), self.create_url_display(item["url"]), f"{confidence:.2f}", item["reason"])
                
                self.console.print(reject_table)
                
                # Option to view full URLs
                if Confirm.ask("View full URLs before deciding?"):
                    self.view_full_urls([item["url"] for item in reject_links], "Rejected Links (Full URLs)")
                
                # Ask which to move
                to_keep = Prompt.ask("Enter numbers to move to KEEP (comma-separated), or 'all' for all")
                
                if to_keep.lower() == 'all':
                    # Move all rejected links to keep
                    for item in reject_links:
                        # Remember this override for future learning
                        self.remember_manual_override(item["url"], "REJECT", "KEEP")
                        
                        # Add to keep_links with updated information
                        keep_item = item.copy()
                        keep_item["reason"] = item["reason"] + " (manually overridden)"
                        keep_item["confidence"] = 1.0  # High confidence for manual decision
                        keep_links.append(keep_item)
                        
                        # Update the decision cache
                        url = item["url"]
                        self.decision_cache[url] = {
                            "decision": "KEEP",
                            "reason": item["reason"] + " (manually overridden)",
                            "source": "manual",
                            "confidence": 1.0
                        }
                        # Update global tracking
                        self.all_keep_urls.add(url)
                        if url in self.all_reject_urls:
                            self.all_reject_urls.remove(url)
                else:
                    # Move selected links
                    try:
                        indices = [int(idx.strip()) - 1 for idx in to_keep.split(",")]
                        for idx in indices:
                            if 0 <= idx < len(reject_links):
                                # Remember this override for future learning
                                self.remember_manual_override(reject_links[idx]["url"], "REJECT", "KEEP")
                                
                                # Add to keep_links with updated information
                                keep_item = reject_links[idx].copy()
                                keep_item["reason"] = reject_links[idx]["reason"] + " (manually overridden)"
                                keep_item["confidence"] = 1.0  # High confidence for manual decision
                                keep_links.append(keep_item)
                                
                                # Update the decision cache
                                url = reject_links[idx]["url"]
                                self.decision_cache[url] = {
                                    "decision": "KEEP",
                                    "reason": reject_links[idx]["reason"] + " (manually overridden)",
                                    "source": "manual",
                                    "confidence": 1.0
                                }
                                # Update global tracking
                                self.all_keep_urls.add(url)
                                if url in self.all_reject_urls:
                                    self.all_reject_urls.remove(url)
                                    
                                self.console.print(f"[green]Moved to KEEP: {reject_links[idx]['url']}[/green]")
                    except ValueError:
                        self.console.print("[yellow]Invalid input. No changes made.[/yellow]")
            else:
                self.console.print("[yellow]No rejected links to move.[/yellow]")
                
        elif choice == "2":
            # Show kept links
            if keep_links:
                keep_table = Table(title="Links Currently Kept")
                keep_table.add_column("#", style="dim", width=4)
                keep_table.add_column("URL", style="green")
                keep_table.add_column("Confidence", width=10)
                keep_table.add_column("Reason")
                
                for idx, item in enumerate(keep_links):
                    confidence = item.get("confidence", 0.5)
                    keep_table.add_row(str(idx+1), self.create_url_display(item["url"]), f"{confidence:.2f}", item["reason"])
                
                self.console.print(keep_table)
                
                # Option to view full URLs
                if Confirm.ask("View full URLs before deciding?"):
                    self.view_full_urls([item["url"] for item in keep_links], "Kept Links (Full URLs)")
                
                # Ask which to move
                to_reject = Prompt.ask("Enter numbers to move to REJECT (comma-separated)")
                
                # Create new keep_links without the rejected ones
                new_keep_links = []
                try:
                    indices = [int(idx.strip()) - 1 for idx in to_reject.split(",")]
                    for idx, item in enumerate(keep_links):
                        if idx in indices:
                            url = item["url"]
                            # Remember this override for future learning
                            self.remember_manual_override(url, "KEEP", "REJECT")
                            
                            # Add to reject_links
                            reject_item = item.copy()
                            reject_item["reason"] = item["reason"] + " (manually overridden)"
                            reject_item["confidence"] = 1.0  # High confidence for manual decision
                            reject_links.append(reject_item)
                            
                            # Update the decision cache
                            self.decision_cache[url] = {
                                "decision": "REJECT",
                                "reason": item["reason"] + " (manually overridden)",
                                "source": "manual",
                                "confidence": 1.0
                            }
                            # Update global tracking
                            self.all_reject_urls.add(url)
                            if url in self.all_keep_urls:
                                self.all_keep_urls.remove(url)
                                
                            self.console.print(f"[red]Moved to REJECT: {item['url']}[/red]")
                        else:
                            new_keep_links.append(item)
                    
                    keep_links = new_keep_links
                except ValueError:
                    self.console.print("[yellow]Invalid input. No changes made.[/yellow]")
            else:
                self.console.print("[yellow]No kept links to move.[/yellow]")
                
        elif choice == "3":
            # View full URLs for all links
            self.console.print("\n[bold]All URLs from this page:[/bold]")
            all_urls = [link["url"] for link in all_links]
            self.view_full_urls(all_urls, "All URLs from this page")
            
            # After viewing, ask if they want to make changes
            if Confirm.ask("Make changes to links now?"):
                # Recursive call to modify links
                return self.modify_page_links(keep_links, reject_links, page_url)
        
        return keep_links

    def manually_review_links(self):
        """Allow manual review and editing of the clean links."""
        # Load the clean links
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                clean_links_data = json.load(f)
            clean_links = clean_links_data.get("clean_links", [])
        except FileNotFoundError:
            self.console.print("[bold red]Clean links file not found. Run process_all_pages() first.[/bold red]")
            return []
        
        # Display the clean links
        self.console.print(f"\n[bold]Final Review of All Clean Links ({len(clean_links)})[/bold]")
        
        # Ask if they want to see the full list or just make changes
        if Confirm.ask("Would you like to view the complete list of clean links?"):
            # Show all links in a table with pagination
            links_per_page = 20
            total_pages = (len(clean_links) + links_per_page - 1) // links_per_page
            
            current_page = 1
            while current_page <= total_pages:
                start_idx = (current_page - 1) * links_per_page
                end_idx = min(start_idx + links_per_page, len(clean_links))
                
                table = Table(title=f"Clean Links (Page {current_page}/{total_pages})")
                table.add_column("#", style="dim")
                table.add_column("URL")
                
                for i in range(start_idx, end_idx):
                    table.add_row(str(i+1), self.create_url_display(clean_links[i]))
                
                self.console.print(table)
                
                # Option to view full URLs for this page
                if Confirm.ask("View full URLs for this page?"):
                    self.view_full_urls(clean_links[start_idx:end_idx], f"Full URLs (Page {current_page})")
                
                if current_page < total_pages:
                    if Confirm.ask("View next page?"):
                        current_page += 1
                    else:
                        break
                else:
                    break
        
        # Ask if user wants to modify the list
        if Confirm.ask("Do you want to modify this final list?"):
            # Options for modification
            self.console.print("\n[bold]Options:[/bold]")
            self.console.print("1. Remove links (provide comma-separated numbers)")
            self.console.print("2. Add new links (provide comma-separated URLs)")
            self.console.print("3. Save and exit")
            
            while True:
                choice = input("\nEnter your choice (1-3): ")
                
                if choice == "1":
                    # Remove links
                    indices = input("Enter numbers to remove (comma-separated): ")
                    try:
                        indices = [int(idx.strip()) - 1 for idx in indices.split(",")]
                        indices.sort(reverse=True)  # Sort in reverse to avoid index shifts
                        
                        for idx in indices:
                            if 0 <= idx < len(clean_links):
                                removed = clean_links.pop(idx)
                                # Remember this override for future learning
                                self.remember_manual_override(removed, "KEEP", "REJECT")
                                
                                # Update global tracking
                                if removed in self.all_keep_urls:
                                    self.all_keep_urls.remove(removed)
                                self.all_reject_urls.add(removed)
                                
                                # Update decision cache
                                self.decision_cache[removed] = {
                                    "decision": "REJECT",
                                    "reason": "Manually removed in final review",
                                    "source": "manual",
                                    "confidence": 1.0
                                }
                                
                                self.console.print(f"Removed: {removed}")
                            else:
                                self.console.print(f"[yellow]Invalid index: {idx+1}[/yellow]")
                    except ValueError:
                        self.console.print("[bold red]Invalid input. Please enter comma-separated numbers.[/bold red]")
                
                elif choice == "2":
                    # Add new links
                    new_links = input("Enter new URLs to add (comma-separated): ")
                    for link in new_links.split(","):
                        link = link.strip()
                        if link and link not in clean_links:
                            clean_links.append(link)
                            # Remember this override for future learning
                            if link in self.all_reject_urls:
                                self.remember_manual_override(link, "REJECT", "KEEP")
                            
                            # Update global tracking
                            self.all_keep_urls.add(link)
                            if link in self.all_reject_urls:
                                self.all_reject_urls.remove(link)
                            
                            # Update decision cache
                            self.decision_cache[link] = {
                                "decision": "KEEP",
                                "reason": "Manually added in final review",
                                "source": "manual",
                                "confidence": 1.0
                            }
                            
                            self.console.print(f"Added: {link}")
                
                elif choice == "3":
                    # Save and exit
                    break
                
                else:
                    self.console.print("[bold red]Invalid choice. Please enter 1, 2, or 3.[/bold red]")
            
            # Save the modified list
            clean_links_data["clean_links"] = clean_links
            clean_links_data["total_count"] = len(clean_links)
            clean_links_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(clean_links_data, f, indent=2)
            
            # Save the updated cache and override patterns
            self.save_decision_cache()
            self.save_override_patterns()
            
            self.console.print(f"\n[bold green]Saved {len(clean_links)} clean links to {self.output_file}[/bold green]")
        
        return clean_links

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Evaluate links using Gemini API with learning from human overrides")
    parser.add_argument("--links-file", required=True, help="Path to the links JSON file from the scraper")
    parser.add_argument("--api-key", required=True, help="Google Gemini API key")
    parser.add_argument("--output", default="clean_links.json", help="Output file for clean links")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze links without generating clean list")
    parser.add_argument("--review", action="store_true", help="Review the results after analysis")
    parser.add_argument("--cache", default="gemini_decisions_cache.json", help="File to store cached decisions")
    parser.add_argument("--overrides", default="override_patterns.json", help="File to store learned override patterns")
    parser.add_argument("--reset-patterns", action="store_true", help="Reset learned patterns before starting")
    
    args = parser.parse_args()
    
    # Initialize the evaluator
    evaluator = LinkEvaluator(
        links_file=args.links_file,
        gemini_api_key=args.api_key,
        output_file=args.output,
        decision_cache_file=args.cache,
        override_patterns_file=args.overrides
    )
    
    # Reset patterns if requested
    if args.reset_patterns:
        evaluator.override_patterns = {}
        evaluator.save_override_patterns()
        print("Learned patterns have been reset.")
    
    # Process the links
    if args.analyze_only:
        evaluator.load_links()
        evaluator.prefilter_links()
        for i in range(len(evaluator.pages)):
            evaluator.analyze_links_with_gemini(i)
    else:
        # Start with option to reset patterns
        evaluator.start_evaluation()
    
    # Display the results for review if requested
    if args.review:
        evaluator.display_gemini_analysis()
        evaluator.manually_review_links()

if __name__ == "__main__":
    main()