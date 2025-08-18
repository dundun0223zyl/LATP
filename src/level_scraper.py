import asyncio
import os
import json
import random
import requests
import time
import re
import base64
import shutil
import argparse
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup
from datetime import datetime
from playwright.async_api import async_playwright
import tldextract  # For better domain handling

class LevelScraper:
    def __init__(self, seed_urls, max_depth=1, output_folder="scraper_output", 
                 min_delay=1.5, max_delay=3.0, request_timeout=30, max_retries=3,
                 visited_urls=None, normalized_visited_urls=None, normalized_base_urls=None):
        """
        Initialize the scraper with configurable parameters.
        
        Args:
            seed_urls (list): List of starting URLs to crawl
            max_depth (int): Maximum crawl depth (default: 1)
            output_folder (str): Main output folder
            min_delay (float): Minimum delay between requests in seconds
            max_delay (float): Maximum delay between requests in seconds
            request_timeout (int): Request timeout in seconds
            max_retries (int): Maximum number of retries for a request
            visited_urls (set): Optional pre-populated set of visited URLs
            normalized_visited_urls (set): Optional pre-populated set of normalized URLs
            normalized_base_urls (set): Optional pre-populated set of normalized base URLs
        """
        # Increased timeouts specifically for government websites
        self.gov_request_timeout = 60  # 60 seconds for government sites
        self.gov_download_timeout = 120  # 2 minutes for government file downloads
        self.gov_wait_time = 5  # 5 seconds wait time for government pages
        self.page_processing_timeout = 180  # 3 minutes maximum for processing any single page
        self.content_extraction_timeout = 30  # 30 seconds for content extraction operations
        
        # Initialize tracking for downloaded files
        self.downloaded_files = set()
        self.download_attempts = set()
        
        # Configuration
        self.seed_urls = seed_urls
        self.max_depth = max_depth
        self.output_folder = output_folder
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        
        # Flag to control folder structure
        self.use_direct_folder_structure = False
        
        # Extract base domain info for the first seed URL (for internal link detection)
        self.base_url = seed_urls[0] if seed_urls else ""
        self.base_domain = urlparse(self.base_url).netloc
        self.base_domain_no_www = self.base_domain.replace('www.', '')
        
        # Special handling for .gov.uk domains
        self.is_gov_domain = '.gov.uk' in self.base_domain
        self.authority = None
        
        if self.is_gov_domain:
            self.authority = self.extract_authority_from_govuk(self.base_domain)
            print(f"Detected gov.uk domain with authority: {self.authority}")
        
        # Define possible variations of the internal domain
        self.internal_domains = [
            self.base_domain,
            self.base_domain_no_www,
            f"www.{self.base_domain_no_www}"
        ]
        
        # For gov.uk domains, add more possible variations
        if self.is_gov_domain and self.authority:
            self.internal_domains.extend([
                f"{self.authority}.gov.uk",
                f"www.{self.authority}.gov.uk"
            ])
            
            # Add common subdomain patterns seen on local authority sites
            common_subdomains = ['services', 'online', 'myaccount', 'secure', 'portal', 'info']
            for subdomain in common_subdomains:
                self.internal_domains.append(f"{subdomain}.{self.authority}.gov.uk")
                
            print(f"Internal domains for {self.authority}: {self.internal_domains}")
        
        self.content_hashes = set()
        
        # File extensions to download
        self.file_extensions = [
            # Documents
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.csv',
            # Images
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp',
            # Other media
            '.mp4', '.mp3', '.wav',
            # Code files
            '.r', '.R', '.Rmd', '.rmd', '.py', '.js', '.cpp', '.h'
        ]
        
        # File download URL patterns - for URLs that don't have standard extensions
        self.file_download_patterns = [
            r"/downloads/file/\d+/",  # Common pattern in gov.uk sites: /downloads/file/12345/filename
            r"/download\.php",
            r"/dl\.php",
            r"/files/",
            r"/get/",
            r"/media/",
            r"/documents/",
            r"/pdfs/",
            r"/publications/",
            r"/attachments/"
        ]
        
        # Content selectors - broader to catch more content across different site structures
        self.content_selectors = {
            # Main content selectors - these look good already
            "main_content": "main, #main, #content, .content, article, [role='main'], .page-content, [class*='container'], [class*='content'], .widget-row, .widget-content",
    
            # Exclude selectors - keep less aggressive
            "exclude": "footer, .footer, .site-footer, #footer, nav.breadcrumb, .breadcrumb, [class*='breadcrumb'], [class*='site-social-bar']",
    
            # Standard selectors - these are fine
            "headings": "h1, h2, h3, h4, h5, h6",
            "paragraphs": "p",
            "lists": "ul, ol",
    
            # Links selector - EXPAND THIS to capture more link types
            "links": "a[href], [role='button'][href], .card a, .tile a, article a, [class*='content'] a, [class*='col'] a, h2 a, h3 a, h4 a, [class*='list'] a, [class*='block'] a",
    
            # Images selector - this is fine
            "images": "img, svg, figure, .figure, [role='img']",
    
            # Header elements - MODIFY THIS to be more selective
            "header_elements": ".site-header, #site-header, .global-header, .site-branding, .main-header, .global-header, .utility-header",
    
            # Footer elements - this is fine
            "footer_elements": "footer, .footer, .site-footer, #footer"
        }
        
        # Patterns to exclude from crawling (navigation, etc.)
        self.exclude_patterns = [
            # Admin and system URLs
            r"/wp-admin/", 
            r"/wp-login",
            r"/#respond", 
            r"/#comment",
            r"/search\?", 
            r"/login",
            r"/signin",
            r"/logout",
            r"/accounts/",
            
            # Asset files
            r"/wp-content/uploads/",  # Skip direct media links
            r"\.(css|js|woff|ttf|ico)$",  # Skip assets
            
            # Fragment URLs
            r"/#.*$"  # Skip all fragment URLs
        ]
        
        # MODIFIED: Initialize tracking variables with provided sets or empty sets
        self.visited_urls = visited_urls if visited_urls is not None else set()
        self.visited_base_urls = set()  # We'll still populate this one locally
        
        # Initialize normalized tracking with provided sets or empty sets
        self.normalized_visited_urls = normalized_visited_urls if normalized_visited_urls is not None else set()
        self.normalized_base_urls = normalized_base_urls if normalized_base_urls is not None else set()
        
        # Print initial tracking info
        print(f"Starting with {len(self.visited_urls)} pre-visited URLs")
        print(f"Starting with {len(self.normalized_visited_urls)} pre-visited normalized URLs")
        
        # Stats for tracking progress and results
        self.stats = {
            "pages_visited": 0,
            "files_downloaded": 0,
            "links_found": 0,
            "internal_links_found": 0,
            "fragment_links_count": 0,
            "level_stats": {f"level_{i}": {"pages": 0, "files": 0} for i in range(self.max_depth + 1)}
        }
        
        # Collected links for evaluation
        self.all_collected_links = []


    def extract_authority_from_govuk(self, domain):
        """
        Extract the authority name from a gov.uk domain.
        Examples:
        - www.coventry.gov.uk -> coventry
        - services.essex.gov.uk -> essex
        - www.cityoflondon.gov.uk -> cityoflondon
        - durham.gov.uk -> durham
        """
        if not domain or '.gov.uk' not in domain:
            return None
        
        # Split by '.gov.uk' to get everything before it
        parts = domain.split('.gov.uk')[0].split('.')
        
        # Handle different patterns
        if len(parts) == 1:
            # Case: durham.gov.uk
            return parts[0]
        elif len(parts) == 2:
            # Case: www.durham.gov.uk or services.durham.gov.uk
            if parts[0] == 'www':
                return parts[1]
            else:
                # Case: services.durham.gov.uk
                return parts[1]  # Assume the authority is the second part
        elif len(parts) >= 3:
            # Case: something.www.durham.gov.uk or similar complex structure
            # Typically, the authority is the last part before .gov.uk
            return parts[-1]
        
        return None

    def normalize_url(self, url):
        """More aggressive URL normalization to catch duplicates."""
        parsed = urlparse(url)
        
        # Get domain without www prefix
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Get path without trailing slash
        path = parsed.path.lower()
        if path.endswith('/'):
            path = path[:-1]
        
        # Sort query parameters for consistent ordering
        query_params = []
        if parsed.query:
            # Parse query string
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            # Sort parameters by name
            for key in sorted(params.keys()):
                # Skip tracking parameters
                if key.lower() in ['utm_source', 'utm_medium', 'utm_campaign', 'source', 'ref']:
                    continue
                # Add all other parameters
                for value in sorted(params[key]):
                    query_params.append(f"{key}={value}")
        
        # Rebuild query string
        query_string = '&'.join(query_params)
        
        # Return normalized URL
        if query_string:
            return f"{domain}{path}?{query_string}"
        else:
            return f"{domain}{path}"

    def get_safe_filename(self, url):
        """Convert a URL to a safe filename."""
        # Remove the fragment part for filenames
        url_without_fragment = urldefrag(url)[0]
        parsed = urlparse(url_without_fragment)
        path = parsed.netloc + parsed.path
        safe_str = re.sub(r'[\\/*?:"<>|]', '_', path)
        safe_str = re.sub(r'_+', '_', safe_str)
        return safe_str[:100] if len(safe_str) > 100 else safe_str

    def is_internal_url(self, url):
        """Check if a URL is internal (hosted on the same website or subdomains)."""
        try:
            parsed_url = urlparse(url)
            
            # If no domain (relative URL), it's internal
            if not parsed_url.netloc:
                return True
            
            # Handle gov.uk domains specially since they have a unique structure
            if self.is_gov_domain:
                # Extract the authority name from the URL being checked
                url_authority = self.extract_authority_from_govuk(parsed_url.netloc)
                
                # If they share the same authority, they're internal to each other
                if self.authority and url_authority and self.authority == url_authority:
                    # Debug output
                    # print(f"✅ Internal gov.uk URL: {url} (authority: {url_authority})")
                    return True
            
            # Standard domain checks for non-gov.uk domains or as fallback
            # Check if the netloc matches any of our internal domains
            if parsed_url.netloc in self.internal_domains:
                # print(f"✅ Internal URL (exact match): {url}")
                return True
            
            # Check if this is a subdomain of our main domain
            for domain in self.internal_domains:
                if parsed_url.netloc.endswith('.' + domain):
                    # print(f"✅ Internal URL (subdomain): {url}")
                    return True
            
            # Handle WordPress CDN URLs (i0.wp.com, i1.wp.com, etc.)
            if parsed_url.netloc.endswith('.wp.com') and self.base_domain_no_www in parsed_url.path:
                # print(f"✅ Internal URL (WordPress CDN): {url}")
                return True
                
            # Debug output
            # print(f"❌ External URL: {url}")
            return False
        except Exception as e:
            print(f"Error checking internal URL {url}: {str(e)}")
            return False

    def should_skip_url(self, url):
        """Check if a URL should be skipped (navigation, etc.)."""
        # Skip non-HTTP URLs
        if not url.startswith(('http://', 'https://')):
            return True
            
        # Skip URLs matching exclude patterns
        for pattern in self.exclude_patterns:
            if re.search(pattern, url):
                return True
        
        # Don't skip if it's a file to download
        if self.is_file_url(url):
            return False
        
        # Default: don't skip
        return False

    def is_file_url(self, url):
        """
        Check if a URL is likely to be a file download link.
        This method checks both file extensions and common file download URL patterns.
        """
        # Check if URL ends with a known file extension
        if any(url.lower().endswith(ext) for ext in self.file_extensions):
            return True
            
        # Check if URL matches any file download patterns
        for pattern in self.file_download_patterns:
            if re.search(pattern, url):
                return True
                
        # Special handling for .gov.uk domains with /downloads/file/ pattern
        if self.is_gov_domain and '/downloads/file/' in url:
            return True
            
        return False

    def is_govuk_download_url(self, url):
        """Check if a URL is a gov.uk download URL that needs special handling."""
        if not url:
            return False
        
        govuk_download_patterns = [
            r"/downloads/file/\d+/",  # Common gov.uk file pattern: /downloads/file/12345/filename
            r"/download/\d+/",        # Alternative download pattern
            r"/documents/\d+/",       # Documents pattern
            r"/publications/\d+/",    # Publications pattern
            r"/attachment/\d+/",      # Attachments pattern
            r"/media/\d+/"            # Media pattern
        ]
        
        # Check if it's a gov.uk domain
        if not self.is_gov_domain and not '.gov.uk' in url:
            return False
        
        # Check if it matches any of the download patterns
        for pattern in govuk_download_patterns:
            if re.search(pattern, url):
                return True
        
        # Check extensions for gov.uk domains
        if any(url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.csv', '.zip']):
            return True
        
        return False

    def get_file_type_from_url(self, url):
        """Try to determine the file type from the URL."""
        # First check for common extensions
        for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.csv', '.zip', '.png', '.jpg', '.jpeg']:
            if url.lower().endswith(ext):
                return ext[1:]  # Return without the dot
        
        # For gov.uk URLs with patterns but no extension
        if self.is_govuk_download_url(url):
            # Default to PDF for government downloads without extension
            return 'pdf'
        
        # Default to binary if we can't determine
        return 'bin'

    def normalize_govuk_url(self, url):
        """Normalize a gov.uk URL for consistent comparison and tracking."""
        # Remove trailing slashes
        url = url.rstrip('/')
        
        # Handle gov.uk download URLs
        if '/downloads/file/' in url:
            # Extract the file ID and name
            match = re.search(r'/downloads/file/(\d+)/([^/]+)', url)
            if match:
                file_id = match.group(1)
                file_name = match.group(2)
                # Normalize to a consistent format
                domain = urlparse(url).netloc
                return f"https://{domain}/downloads/file/{file_id}/{file_name}"
        
        return url
    
    #def get_content_hash(self, soup):
       # """Generate a content hash based on main content rather than entire page."""
        # Try to find main content container
        #main_content = None
        #for selector in "main, #main, .site-main, article, [role='main'], #content, .content":
            #elements = soup.select(selector)
            #if elements:
                #main_content = elements[0]
                #break

        ## If no main content found, use body with headers/footers removed
        #if not main_content:
            #main_content = soup.body
            # Remove headers, footers, navs
            #for selector in "header, footer, nav, .header, .footer, .nav":
                #for element in main_content.select(selector):
                    #element.decompose()

        # Get text content, normalize whitespace and generate hash
        #content_text = main_content.get_text(separator=' ', strip=True)
        # Remove excess whitespace and convert to lowercase for more robust comparison
        #normalized_content = ' '.join(content_text.lower().split())
        # Use only first 2000 chars for faster comparison while still being distinctive
        #content_sample = normalized_content[:2000]
        # Create hash from content sample
        #import hashlib
        #return hashlib.md5(content_sample.encode('utf-8')).hexdigest()

    def verify_downloaded_file(self, file_path):
        """Verify a downloaded file exists and has content."""
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "File is empty (0 bytes)"
        
        return True, f"File exists with size: {file_size} bytes"

    def save_download_summary(self, level_folder, media_links, downloaded_files):
        """
        Save a summary of download attempts and results.
        
        Args:
            level_folder: The folder for the current level
            media_links: List of all links that were attempted
            downloaded_files: List of dictionaries with download results
        """
        download_summary = {
            "total_attempted": len(media_links),
            "successful": len([f for f in downloaded_files if f["success"]]),
            "failed": len([f for f in downloaded_files if not f["success"]]),
            "details": downloaded_files
        }
        
        summary_path = os.path.join(level_folder, "download_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(download_summary, f, indent=2)
        
        print(f"\n===== DOWNLOAD SUMMARY =====")
        print(f"Attempted: {download_summary['total_attempted']}")
        print(f"Successful: {download_summary['successful']}")
        print(f"Failed: {download_summary['failed']}")
        print(f"Full details saved to: {summary_path}")
        
        return download_summary

    async def download_file_with_playwright(self, page, file_url, file_path, timeout=30000, source_page_url=None):
        """
        Download a file using Playwright with configurable timeout.
        This is useful for files that require JavaScript or session cookies to download.
        Returns a tuple of (success, fallback_needed)
        
        Args:
            page: Playwright page object
            file_url: URL of the file to download
            file_path: Path where the file should be saved
            timeout: Timeout in milliseconds (default: 30000)
        """
        try:
            print(f"Using Playwright for government file: {file_url}")
            
            # Create a download promise before navigation
            download_promise = asyncio.create_task(self.setup_download_listener(page))
            
            # Go to the URL with longer timeout for government sites
            try:
                print(f"Navigating to {file_url} with {timeout}ms timeout")
                response = await page.goto(
                    file_url,
                    timeout=timeout,  # Use the configurable timeout
                    wait_until="domcontentloaded"
                )
                
                if not response or response.status >= 400:
                    print(f"Playwright navigation returned status {response.status if response else 'None'}")
                    print(f"This is normal for file downloads - falling back to requests method...")
                    return False, True  # Not successful, but need fallback
                    
            except Exception as e:
                # Check if this is the common ERR_ABORTED error which is actually normal for downloads
                if "net::ERR_ABORTED" in str(e):
                    print(f"Got expected net::ERR_ABORTED error - this is normal for government file downloads")
                    # Continue execution - don't return yet as the download might still be in progress
                else:
                    print(f"Playwright navigation error: {str(e)}")
                    print(f"Falling back to requests method...")
                    return False, True  # Not successful, but need fallback
                
            # Wait for the download to start (up to 15 seconds for government sites)
            try:
                # Longer wait time for government sites
                download_timeout = 15.0 if self.is_gov_domain else 10.0
                print(f"Waiting up to {download_timeout} seconds for download to start...")
                
                download = await asyncio.wait_for(download_promise, timeout=download_timeout)
                if download:
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Save the file with a longer timeout for government sites
                    save_timeout = 60.0 if self.is_gov_domain else 30.0
                    print(f"Download started, saving with {save_timeout}s timeout...")
                    
                    # Use a separate task with timeout for saving
                    save_task = asyncio.create_task(download.save_as(file_path))
                    try:
                        await asyncio.wait_for(save_task, timeout=save_timeout)
                    except asyncio.TimeoutError:
                        print(f"Timeout saving file after {save_timeout} seconds")
                        return False, True  # Failed, try fallback
                    
                    # Verify the file exists and has size
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        print(f"Successfully downloaded {file_url} using Playwright to {file_path}")
                        return True, False  # Success, no fallback needed
                    else:
                        print(f"File appears to be empty or wasn't saved properly: {file_path}")
                        return False, True  # Failed, try fallback
                else:
                    print(f"No download started for {file_url}")
            except asyncio.TimeoutError:
                print(f"Timeout waiting for download to start for {file_url}")
            
            # If we get here without returning, try clicking download links as a last Playwright attempt
            try:
                # Look for likely download buttons/links
                download_selectors = [
                    'a[href*="download"]', 
                    'a[href$=".pdf"]', 
                    'button:has-text("Download")', 
                    'a:has-text("Download")',
                    'a.download-link',
                    'a[download]',
                    '.btn-download',  # Added more selectors for government sites
                    '[role="button"]',
                    'input[type="submit"]',
                    'button[type="submit"]'
                ]
                
                for selector in download_selectors:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        for element in elements:
                            try:
                                # Setup a new download listener
                                click_download_promise = asyncio.create_task(self.setup_download_listener(page))
                                
                                # Click the element
                                await element.click()
                                
                                # Wait for download
                                try:
                                    download = await asyncio.wait_for(click_download_promise, timeout=download_timeout)
                                    if download:
                                        await download.save_as(file_path)
                                        
                                        # Verify the file exists and has size
                                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                                            print(f"Successfully downloaded via click on {selector}")
                                            return True, False  # Success, no fallback needed
                                        else:
                                            print(f"File appears to be empty after clicking {selector}")
                                except asyncio.TimeoutError:
                                    print(f"Timeout waiting for download after clicking {selector}")
                            except Exception as click_error:
                                print(f"Error clicking {selector}: {str(click_error)}")
            
            except Exception as e:
                print(f"Error looking for download buttons: {str(e)}")
            
            # As a last resort, check if the page content itself is the file
            try:
                content_type = await self.safe_page_operation(
                    page.evaluate("""() => {
                        return document.contentType || document.mimeType || '';
                    }"""),
                    "content type detection"
                )
                
                if content_type and 'application/' in content_type and not 'text/html' in content_type:
                    print(f"Detected non-HTML content type: {content_type}")
                    
                    # Get the page content
                    content = await self.safe_page_operation(page.content(), "page content extraction")
                    
                    # Save the content as a file
                    with open(file_path, 'wb') as f:
                        f.write(content.encode('utf-8'))
                    
                    # Verify
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        print(f"Saved page content as file: {file_path}")
                        return True, False
            except Exception as content_error:
                print(f"Error saving page content: {str(content_error)}")
            
            # If we get here, the Playwright download failed but we should try the requests method
            return False, True  # Failed, try fallback
                
        except Exception as e:
            print(f"Error in download_file_with_playwright: {str(e)}")
            return False, True  # Failed, try fallback

    def download_file(self, file_url, file_path, timeout=30, source_page_url=None):
        """
        Download a file with retries using requests with improved error handling.
        Includes special handling for government websites and source tracking.
        
        Args:
            file_url: URL of the file to download
            file_path: Path where the file should be saved
            timeout: Request timeout in seconds (default: 30)
            source_page_url: URL of the page where this file link was found
        """
        max_retries = self.max_retries
        for retry in range(max_retries):
            try:
                # Add proper headers that mimic a browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/pdf',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': self.base_url,
                    'DNT': '1',  # Do Not Track
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                # Use session to maintain cookies
                session = requests.Session()
                
                # First, make a HEAD request to check if the file exists and follow redirects
                try:
                    print(f"Making HEAD request to {file_url} with {timeout}s timeout")
                    head_response = session.head(
                        file_url, 
                        timeout=timeout, 
                        headers=headers, 
                        allow_redirects=True
                    )
                    
                    # If we got redirected, use the final URL
                    if head_response.url != file_url:
                        print(f"Redirected from {file_url} to {head_response.url}")
                        file_url = head_response.url
                        
                    # Check for content type from HEAD
                    content_type = head_response.headers.get('Content-Type', '')
                    content_length = head_response.headers.get('Content-Length', '0')
                    print(f"Content-Type: {content_type}, Content-Length: {content_length}")
                    
                except Exception as e:
                    print(f"HEAD request failed for {file_url}: {str(e)}")
                    print("Continuing with original URL...")
                
                # Now make the actual GET request to download the file
                print(f"Making GET request to download {file_url}")
                response = session.get(
                    file_url, 
                    stream=True, 
                    timeout=timeout, 
                    headers=headers, 
                    allow_redirects=True
                )
                
                # Check for successful response
                if response.status_code == 200:
                    # Check content type to ensure it's actually a file
                    content_type = response.headers.get('Content-Type', '')
                    content_length = response.headers.get('Content-Length', '0')
                    print(f"Download response: {response.status_code}, Content-Type: {content_type}, Content-Length: {content_length}")
                    
                    # Handle missing extensions based on content type
                    if '.' not in file_path or file_path.endswith('.bin'):
                        if 'application/pdf' in content_type:
                            if not file_path.lower().endswith('.pdf'):
                                file_path = file_path + '.pdf'
                        elif 'application/msword' in content_type or 'application/vnd.openxmlformats-officedocument.wordprocessingml' in content_type:
                            if not file_path.lower().endswith(('.doc', '.docx')):
                                file_path = file_path + '.docx'
                        elif 'application/vnd.ms-excel' in content_type or 'application/vnd.openxmlformats-officedocument.spreadsheetml' in content_type:
                            if not file_path.lower().endswith(('.xls', '.xlsx')):
                                file_path = file_path + '.xlsx'
                    
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Save the file with progress tracking for large files
                    file_size = int(content_length) if content_length.isdigit() else 0
                    downloaded = 0
                    
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  # Filter out keep-alive chunks
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # Print progress for large files
                                if file_size > 1000000:  # 1MB
                                    percent = (downloaded / file_size) * 100 if file_size > 0 else 0
                                    print(f"\rDownloading: {downloaded/1024/1024:.1f} MB / {file_size/1024/1024:.1f} MB ({percent:.1f}%)", end="")
                    
                    if file_size > 1000000:  # 1MB
                        print()  # New line after progress
                    
                    # Verify the file exists and has content
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        actual_size = os.path.getsize(file_path)
                        print(f"Successfully downloaded {file_url} to {file_path} ({actual_size} bytes)")
                        
                        # Store this URL in global downloaded_files set for tracking
                        if not hasattr(self, 'downloaded_files'):
                            self.downloaded_files = set()
                        self.downloaded_files.add(file_url)
                        
                        # Add source tracking
                        success = True
                        # Create source mapping file if it doesn't exist
                        source_mapping_file = os.path.join(os.path.dirname(os.path.dirname(file_path)), "file_sources.json")
                        try:
                            if os.path.exists(source_mapping_file):
                                with open(source_mapping_file, 'r') as f:
                                    source_mapping = json.load(f)
                            else:
                                source_mapping = {}
                                
                            # Add this file to the mapping
                            filename = os.path.basename(file_path)
                            source_mapping[filename] = {
                                'source_url': source_page_url or self.base_url,
                                'download_url': file_url,
                                'download_time': datetime.now().isoformat()
                            }
                            
                            # Save updated mapping
                            with open(source_mapping_file, 'w') as f:
                                json.dump(source_mapping, f, indent=2)
                                
                        except Exception as e:
                            print(f"Error updating source mapping: {str(e)}")
                        
                        return True
                    else:
                        print(f"Downloaded file is empty: {file_path}")
                        if retry < max_retries - 1:
                            time.sleep(2 ** retry)  # Exponential backoff
                            continue
                        return False
                else:
                    print(f"Failed to download {file_url}, status: {response.status_code}")
                    if retry < max_retries - 1:
                        time.sleep(2 ** retry)  # Exponential backoff
                        continue
                    return False
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(2 ** retry)  # Exponential backoff
                    print(f"Retry {retry+1}/{max_retries} for file: {file_url}")
                else:
                    print(f"Error downloading {file_url}: {str(e)}")
                    return False
        
        return False

    async def setup_download_listener(self, page):
        """Set up a listener for the download event."""
        try:
            async with page.expect_download() as download_info:
                return await download_info.value
        except Exception as e:
            print(f"Error setting up download listener: {str(e)}")
            return None

    def detect_site_type(self, url):
        """Detect the type of site to apply appropriate extraction strategies."""
        if self.is_internal_url(url):
            return "internal"
        else:
            return "external"  # This should never happen with our filtering

    def debug_html_structure(self, soup):
        """Debug helper to print the HTML structure with focus on headers and footers."""
        try:
            # Print top-level elements
            print("\n=== PAGE STRUCTURE DEBUGGING ===")
            for i, element in enumerate(soup.body.children):
                if hasattr(element, 'name') and element.name:
                    attrs = ' '.join([f'{k}="{v}"' for k, v in element.attrs.items()])
                    print(f"L1: <{element.name} {attrs}>")
                    
                    # Look for headers and footers
                    if element.name in ['header', 'footer'] or any(cls in element.get('class', []) for cls in ['header', 'footer', 'site-header', 'site-footer']):
                        print(f"  [!] Found potential header/footer: {element.name} {attrs}")
                        
                        # Print children of the header/footer
                        for child in element.find_all(['div', 'nav', 'ul']):
                            child_attrs = ' '.join([f'{k}="{v}"' for k, v in child.attrs.items()])
                            print(f"    - <{child.name} {child_attrs}>")
                            
                            # Print links in the header/footer
                            for link in child.find_all('a', href=True):
                                print(f"      * <a href=\"{link['href']}\"> {link.get_text()[:30].strip()} </a>")
            
            # Check for elements with 'footer' or 'header' in their class/id
            print("\n=== HEADER/FOOTER CLASS ELEMENTS ===")
            header_footer_elems = soup.select('[class*="header"], [class*="footer"], [id*="header"], [id*="footer"]')
            for elem in header_footer_elems:
                attrs = ' '.join([f'{k}="{v}"' for k, v in elem.attrs.items()])
                print(f"Element: <{elem.name} {attrs}>")
                
            print("=== END DEBUGGING ===\n")
        except Exception as e:
            print(f"Error in debug_html_structure: {str(e)}")
            
    def debug_links(self, links, label="Links"):
        """Debug helper to print links for analysis."""
        print(f"\n=== {label} ({len(links)}) ===")
        for i, link in enumerate(links[:20]):  # Print up to 20 links
            print(f"{i+1}. {link}")
        if len(links) > 20:
            print(f"... and {len(links) - 20} more")
        print(f"=== END {label} ===\n")

    async def _bypass_with_javascript(self, page):
        """Try to find and click cookie buttons with JavaScript."""
        return await page.evaluate("""() => {
            // Common button texts to look for
            const buttonTexts = ['Accept', 'Accept all', 'I agree', 'Allow all', 'Accept cookies', 
                                 'Yes', 'OK', 'Continue', 'Allow cookies', 'Close'];
            
            // Check all buttons for these texts
            const allButtons = Array.from(document.querySelectorAll('button, a.button, input[type="button"], [role="button"]'));
            for (const button of allButtons) {
                const buttonText = button.innerText || button.value || '';
                for (const text of buttonTexts) {
                    if (buttonText.toLowerCase().includes(text.toLowerCase())) {
                        console.log('Clicking button with text: ' + buttonText);
                        button.click();
                        return true;
                    }
                }
            }
            
            // Common selectors for cookie buttons
            const selectors = [
                '#ccc-button-holder .ccc-notify-accept',
                '.cookie-banner__button--accept',
                '#cookie-accept-all-button',
                '.govuk-cookie-banner button',
                '.govuk-button[data-module="govuk-button"]',
                '.cookie-banner button',
                '#cookie-banner button',
                '[data-cookie-consent] button',
                '.cookie-accept-all',
                '[data-purpose="cookie-banner-ok"]'
            ];
            
            for (const selector of selectors) {
                const element = document.querySelector(selector);
                if (element) {
                    console.log('Found and clicked: ' + selector);
                    element.click();
                    return true;
                }
            }
            
            return false;
        }""")

    async def _bypass_with_frames(self, page):
        """Check for cookie banners in frames/iframes."""
        frames = page.frames
        for frame in frames:
            if frame != page.main_frame:
                try:
                    success = await frame.evaluate("""() => {
                        // Same button logic as in JavaScript bypass, but in frame context
                        const buttonTexts = ['Accept', 'Accept all', 'I agree', 'Allow'];
                        const allButtons = Array.from(document.querySelectorAll('button, a.button'));
                        for (const button of allButtons) {
                            const buttonText = button.innerText || '';
                            for (const text of buttonTexts) {
                                if (buttonText.toLowerCase().includes(text.toLowerCase())) {
                                    button.click();
                                    return true;
                                }
                            }
                        }
                        return false;
                    }""")
                    if success:
                        return True
                except:
                    continue
        return False

    async def _bypass_with_storage(self, page):
        """Try to bypass using sessionStorage."""
        return await page.evaluate("""() => {
            try {
                // Common session storage keys used by cookie consent tools
                sessionStorage.setItem('cookie_consent', 'true');
                sessionStorage.setItem('cookies_accepted', 'true');
                sessionStorage.setItem('cookie_notice_accepted', 'true');
                sessionStorage.setItem('cookie-policy-accepted', 'true');
                return true;
            } catch (e) {
                return false;
            }
        }""")

    async def _bypass_with_localstorage(self, page):
        """Try to bypass using localStorage."""
        return await page.evaluate("""() => {
            try {
                // Common local storage keys used by cookie consent tools
                localStorage.setItem('cookie_consent', 'true');
                localStorage.setItem('cookies_accepted', 'true');
                localStorage.setItem('cookie_notice_accepted', 'true');
                localStorage.setItem('cookie-policy-accepted', 'true');
                localStorage.setItem('civicCookieControl', JSON.stringify({
                    'consentDate': new Date().toISOString(),
                    'optionalCookies': {'analytical': 'accepted', 'marketing': 'accepted'}
                }));
                return true;
            } catch (e) {
                return false;
            }
        }""")
    
    async def improved_cookie_bypass(self, page, base_url):
        """Generalized cookie bypass that works for government and local authority sites."""
        print("Attempting improved cookie consent bypass...")
        
        # Add timeout for cookie bypass attempts
        bypass_timeout = 15  # 15 seconds max for cookie bypass attempts
        start_time = time.time()
        
        try:
            # 1. Try to click accept buttons with flexible matching
            clicked = await page.evaluate("""() => {
                // Look for any button containing "accept" and "cookie" or "all"
                const buttons = Array.from(document.querySelectorAll('button, [role="button"], a.button, input[type="button"]'));
                
                // Try exact button text patterns common in UK government sites
                const commonPatterns = [
                    'accept all',
                    'accept cookies', 
                    'i accept',
                    'accept additional',
                    'accept non-essential',
                    'allow all',  // Added more patterns
                    'agree',
                    'continue',
                    'ok',
                    'yes'
                ];
                
                // First try to find exact matches for common government patterns
                for (const button of buttons) {
                    const text = (button.innerText || button.value || '').toLowerCase().trim();
                    for (const pattern of commonPatterns) {
                        if (text.includes(pattern)) {
                            console.log('Clicking matched button: "' + text + '"');
                            button.click();
                            return true;
                        }
                    }
                }
                
                // Try common selectors in government sites
                const selectors = [
                    '.cookie-banner__button--accept',
                    '.govuk-button[data-module="govuk-button"]',
                    '[class*="cookie"] button',
                    '[class*="cookie"] [role="button"]',
                    '[class*="consent"] button',
                    '#ccc-button-holder .ccc-notify-accept',  // Added more selectors
                    '#cookie-accept-all-button',
                    '.govuk-cookie-banner button',
                    '.cookie-banner button',
                    '#cookie-banner button',
                    '[data-cookie-consent] button',
                    '.cookie-accept-all',
                    '[data-purpose="cookie-banner-ok"]',
                    // Southampton-specific selectors
                    '.scc-cookie-banner-accept',
                    '#cookie-consent-accept',
                    '.scc-cookie-banner button'
                ];
                
                for (const selector of selectors) {
                    const element = document.querySelector(selector);
                    if (element) {
                        console.log('Clicking element with selector: ' + selector);
                        element.click();
                        return true;
                    }
                }
                
                // If specific selectors didn't work, try clicking any button in a cookie banner
                const cookieBanners = document.querySelectorAll('[class*="cookie-banner"], [id*="cookie-banner"], [class*="cookieBanner"]');
                for (const banner of cookieBanners) {
                    const bannerButtons = banner.querySelectorAll('button');
                    for (const btn of bannerButtons) {
                        console.log('Trying to click button in cookie banner');
                        btn.click();
                        return true;
                    }
                }
                
                return false;
            }""")
            
            if clicked:
                print("Successfully clicked cookie consent button")
                await asyncio.sleep(3)  # Wait longer for banner to disappear
            else:
                print("Could not find a matching cookie consent button")
                
                # Try the additional bypass methods if main click didn't work
                if time.time() - start_time < bypass_timeout:
                    print("Trying alternative bypass methods...")
                    try:
                        if await self._bypass_with_javascript(page):
                            print("JavaScript bypass succeeded")
                        elif await self._bypass_with_frames(page):
                            print("Frame bypass succeeded")
                        elif await self._bypass_with_storage(page):
                            print("Session storage bypass succeeded")
                        elif await self._bypass_with_localstorage(page):
                            print("Local storage bypass succeeded")
                    except Exception as e:
                        print(f"Error in alternative bypass methods: {str(e)}")
            
            # 2. Set generic cookie consent values that work across sites
            await page.evaluate("""() => {
                // Set generic cookie consent values
                const cookieValues = [
                    ['cookies_policy', 'true'],
                    ['cookies_preferences_set', 'true'],
                    ['cookie_consent', 'true'],
                    ['cookie_control', 'true'],
                    ['cookie_notice_accepted', 'true'],
                    ['cookie-preferences', 'true'],
                    ['cookie_settings', 'true'],
                    ['essential', 'true'],
                    ['usage', 'true'],
                    ['non-essential-cookies-allowed', 'true'],
                    // Add Southampton-specific cookies
                    ['SCC_COOKIE_BANNER_SHOWN', 'true'],
                    ['scc_cookie_banner_accepted', 'true']
                ];
                
                // Set all common cookie values
                cookieValues.forEach(([name, value]) => {
                    document.cookie = `${name}=${value}; path=/; max-age=31536000`;
                });
                
                // Try to remove any visible banners
                const banners = document.querySelectorAll('[class*="cookie"], [id*="cookie"], [class*="banner"], [aria-label*="cookie"]');
                banners.forEach(banner => {
                    if (banner) {
                        banner.style.display = 'none';
                        banner.setAttribute('aria-hidden', 'true');
                    }
                });
            }""")
            
            # Add timeout check - if we've spent too long already, skip the page reload
            if time.time() - start_time > bypass_timeout:
                print("Cookie bypass timeout reached, continuing without page reload")
                return False
                
            # 3. Critical: Navigate to the page again with cookies set
            print("Navigating to page again with cookies set")
            try:
                await page.goto(base_url, timeout=30000)
                await page.wait_for_load_state('domcontentloaded')
            except Exception as e:
                print(f"Error reloading page after cookie bypass: {str(e)}")
                print("Continuing anyway...")
                return False
            
            return True
        except Exception as e:
            print(f"Error in improved cookie bypass: {str(e)}")
            return False
        finally:
            # Even if we encounter an error, check if we've exceeded timeout
            if time.time() - start_time > bypass_timeout:
                print("Cookie bypass timeout reached")
                return False


    async def safe_page_operation(self, operation, operation_name, timeout=None):
        """Execute a page operation with timeout protection."""
        if timeout is None:
            timeout = self.content_extraction_timeout
        
        try:
            return await asyncio.wait_for(operation, timeout=timeout)
        except asyncio.TimeoutError:
            print(f"⚠️  {operation_name} timeout ({timeout}s) exceeded")
            return None
        except Exception as e:
            print(f"⚠️  Error in {operation_name}: {e}")
            return None

    async def process_page_with_timeout(self, page, url, depth):
        """Process a single page with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.process_page(page, url, depth),
                timeout=self.page_processing_timeout
            )
        except asyncio.TimeoutError:
            print(f"⚠️  Page processing timeout ({self.page_processing_timeout}s) exceeded for: {url}")
            return []
        except Exception as e:
            print(f"⚠️  Error processing page {url}: {e}")
            return []

    async def process_page(self, page, url, depth):
        """Process a single page: extract content, download files, and find links."""
        # Handle fragment identifiers properly
        base_url, fragment = urldefrag(url)
    
        print(f"\n\n==== PROCESSING PAGE: {url} (Level: {depth}/{self.max_depth}) ====")
        if fragment:
            print(f"Fragment identifier detected: #{fragment}")
    
        if self.should_skip_url(url):
            print(f"Skipping URL (matches exclude pattern): {url}")
            return []
    
        # Check if it's an internal URL - only process internal URLs
        if not self.is_internal_url(url):
            print(f"Skipping external URL: {url}")
            return []
    
        # Add a random delay
        await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
    
        # Update stats
        self.stats["pages_visited"] += 1
        self.stats["level_stats"][f"level_{depth}"]["pages"] += 1
    
        # Detect site type for special handling
        site_type = self.detect_site_type(url)
        print(f"Detected site type: {site_type}")
    
        # Use longer timeout for government websites
        page_timeout = self.gov_request_timeout * 1000 if self.is_gov_domain else self.request_timeout * 1000
    
        # Navigate to the page with retries - use base_url without fragment
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Set a user agent to look more like a regular browser
                await page.set_extra_http_headers({
                    # Use a very current Chrome version
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    
                    # More detailed accept headers that real browsers use
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-GB,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
    
                    # These sec- headers are important for modern sites
                    'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"',
                    'sec-fetch-dest': 'document',
                    'sec-fetch-mode': 'navigate',
                    'sec-fetch-site': 'none',
                    'sec-fetch-user': '?1',
    
                    # Other important headers
                    'Upgrade-Insecure-Requests': '1',
                    'Connection': 'keep-alive',
                    'Cache-Control': 'max-age=0'
                })

                # Better timeout handling
                try:
                    # Navigate to the base URL (without fragment)
                    print(f"Navigating to {base_url} with {page_timeout}ms timeout")
                    response = await page.goto(
                        base_url,  # Use base_url without the fragment
                        timeout=page_timeout,  # Use longer timeout for government sites
                        wait_until="domcontentloaded"
                    )
                except Exception as e:
                    print(f"Timeout or error navigating to {base_url}: {str(e)}")
                    return []  # Skip this URL
                
                if not response:
                    raise Exception("No response received")
                    
                status = response.status
                if status >= 400:
                    print(f"Error status code: {status}")
                    
                    if retry < max_retries - 1:
                        await asyncio.sleep(2 ** retry)  # Exponential backoff
                        continue
                    return []
                
                # Wait for initial page load
                await page.wait_for_load_state('networkidle', timeout=10000)
                await asyncio.sleep(2)

                print("Attempting to bypass cookie banners...")

                try:
                    # Use the improved generalized method
                    success = await self.improved_cookie_bypass(page, base_url)
                    if success:
                        print("Successfully bypassed cookie banner")
                    else:
                        print("Cookie bypass was not successful, continuing anyway...")
                        # Attempt a simple force hide of any remaining banners as last resort
                        await page.evaluate("""() => {
                            // Hide any elements that might be cookie banners
                            document.querySelectorAll('[class*="cookie"], [id*="cookie"], [class*="banner"], [class*="consent"]').forEach(el => {
                                if (el) el.style.display = 'none';
                            });
                        }""")
                except Exception as e:
                    print(f"Error handling cookies: {str(e)}")
                    print("Continuing page processing despite cookie error")
    
                    # Additional wait for dynamic content
                    await asyncio.sleep(3)
                
                # Wait longer for government sites to load
                if self.is_gov_domain:
                    print(f"Government site detected - waiting {self.gov_wait_time} seconds for page to stabilize")
                    await asyncio.sleep(self.gov_wait_time)
                else:
                    # Wait for any JS to load and stabilize
                    await asyncio.sleep(3)
                
                # If there was a fragment, try to scroll to it
                if fragment:
                    try:
                        # Try to scroll to the element with that ID
                        await page.evaluate(f"""() => {{
                            const element = document.getElementById('{fragment}');
                            if (element) {{
                                element.scrollIntoView();
                            }} else {{
                                // Try to find by name attribute
                                const namedElement = document.getElementsByName('{fragment}')[0];
                                if (namedElement) {{
                                    namedElement.scrollIntoView();
                                }}
                            }}
                        }}""")
                        # Wait a bit for any effects after scrolling
                        await asyncio.sleep(1)
                    except Exception as e:
                        print(f"Error scrolling to fragment #{fragment}: {str(e)}")
                
                # Wait for common page elements to be available
                try:
                    await page.wait_for_selector('a, p, div, article, #content', timeout=5000)
                except:
                    print("Couldn't find basic page elements, but continuing anyway")
                
                break  # Success, exit retry loop
                
            except Exception as e:
                print(f"Error navigating to {url}: {str(e)}")
                if retry < max_retries - 1:
                    print(f"Retrying ({retry+1}/{max_retries})...")
                    await asyncio.sleep(2 ** retry)  # Exponential backoff
                else:
                    print(f"Failed after {max_retries} attempts")
                    return []
    
        # Process the page content
        try:
            # Get complete HTML content
            try:
                html_content = await self.safe_page_operation(page.content(), "page content extraction")
                if html_content is None:
                    print("⚠️  Could not extract page content, skipping page")
                    return []
                
                # ADD DEBUG CODE HERE - right after getting html_content but before BeautifulSoup
                print("\n===== CONTENT DEBUG =====")
                print(f"Content length: {len(html_content)} characters")
                page_title = await self.safe_page_operation(page.title(), "page title extraction")
                print(f"Page title: {page_title}")
    
                # Check for login pages or minimal content
                if "sign in" in html_content.lower() or "log in" in html_content.lower():
                    print("WARNING: Page appears to be a login page!")
                if len(html_content) < 5000:
                    print("WARNING: Very little content found, page may not have loaded properly")
                
                
                # NEW: Add cookie-specific check here
                if ("essential cookies" in html_content.lower() or 
                    "we use cookies" in html_content.lower() or 
                    "cookie policy" in html_content.lower() or
                    "accept cookies" in html_content.lower()) and len(html_content) < 10000:
                    print("WARNING: Page still shows cookie consent content!")
        
                    # Try to get actual content via JavaScript
                    main_content = await self.safe_page_operation(
                        page.evaluate("""() => {
                            // Try multiple selectors for main content
                            const selectors = ['main', '#main', '#content', '.content', 'article', '.page-content', '[role="main"]'];
                            for (const selector of selectors) {
                                const element = document.querySelector(selector);
                                if (element && element.innerText && element.innerText.length > 100) {
                                    return {
                                        found: true,
                                        selector: selector,
                                        content: element.innerText.substring(0, 500) + '...',
                                        fullLength: element.innerText.length
                                    };
                                }
                            }
                            return {found: false, content: null};
                        }"""),
                        "main content extraction"
                    )
                
                    if main_content and main_content['found']:
                        print(f"Found main content via JavaScript using selector: {main_content['selector']}")
                        print(f"Main content length: {main_content['fullLength']} characters")
                        print(f"Preview: {main_content['content'][:200]}...")
                    else:
                        print("ERROR: Could not find main content - cookie bypass likely failed")
            
                        # Additional diagnostic: check what's visible on the page
                        visible_text = await self.safe_page_operation(
                            page.evaluate("""() => {
                                const body = document.body;
                                return body ? body.innerText.substring(0, 1000) : 'No body content';
                            }"""),
                            "visible text extraction"
                        )
                        if visible_text:
                            print(f"Visible text preview: {visible_text[:300]}...")
                        else:
                            print("Could not extract visible text")

                # Take screenshot to see what's actually rendering
                debug_dir = os.path.join(self.output_folder, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                await self.safe_page_operation(
                    page.screenshot(path=os.path.join(debug_dir, f"{self.get_safe_filename(url)}.png")),
                    "page screenshot"
                )
                print(f"Screenshot saved to debug folder")
                print("========================\n")

            except Exception as e:
                print(f"Error getting page content: {str(e)}")
                return []
            
            # Find all base64 embedded images
            base64_images = []
            base64_elements = await page.query_selector_all('img[src^="data:"]')
            for element in base64_elements:
                src = await element.get_attribute('src')
                if src and src.startswith('data:image/'):
                    alt = await element.get_attribute('alt') or 'embedded-image'
                    base64_images.append((src, alt))
        
            
            # Parse HTML with BeautifulSoup for better content extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Check if we got a valid page
            if not soup or not soup.body:
                print(f"Invalid HTML content received for {url}")
                return []
            
            # ADD CONTENT HASH CHECKING HERE
            #content_hash = self.get_content_hash(soup)
            #if content_hash in self.content_hashes:
                #print(f"Skipping duplicate content: {url} (hash: {content_hash[:8]}...)")
                #return []  # Skip processing this page
    
            # Otherwise add hash to seen set and continue processing
            #self.content_hashes.add(content_hash) 
            
            # ADD CONTENT SELECTION DEBUG HERE
            print("\n=== CONTENT SELECTION DEBUG ===")
            for selector in "main, #main, #content, .content, article, [role='main'], .site-main, .page-content, .site-content, [class*='primary'], [class*='main-content'], .container".split(', '):
                elements = soup.select(selector)
                if elements:
                    print(f"Found {len(elements)} elements with selector '{selector}'")
                    for i, elem in enumerate(elements[:2]):  # Show info for first 2 elements only
                        text_length = len(elem.get_text(strip=True))
                        classes = elem.get('class', [])
                        print(f"  Element {i+1}: <{elem.name}> Classes: {classes}, Text length: {text_length} chars")
            print("===============================\n")
                
            # Optional: Debug the HTML structure to help identify headers and footers
            # Uncomment this line during development to see the structure
            # self.debug_html_structure(soup)
            
            # Remove unwanted elements first
            for selector in self.content_selectors["exclude"].split(','):
                for element in soup.select(selector.strip()):
                    element.decompose()
            
            # Explicitly remove cookie banners before extraction
            #for banner in soup.select('[class*="cookie"], [id*="cookie"], .govuk-cookie-banner, .cookie-banner'):
                #banner.decompose()
            
            # Extract main content
            # Find the main content container using common selectors
            content_text = ""
            main_element = None
            content_containers = []

            # Try to find the main content container using common selectors
            for selector in "main, #main, .site-main, .site-content, .page-primary__content, .main-content, [role='main'], article, .container > .row > div, #content, .content, .page-content".split(', '):
                elements = soup.select(selector)
                for element in elements:
                    # Skip tiny containers and likely navigation elements
                    text = element.get_text(strip=True)
                    if len(text) < 100 or element.name == 'nav' or 'navigation' in str(element.get('class', [])):
                        continue
        
                    # Add to potential containers with priority based on likely relevance
                    priority = 0
                    if element.name == 'main':
                        priority = 10
                    elif 'content' in str(element.get('class', [])) or 'content' in str(element.get('id', '')):
                        priority = 8
                    elif element.name == 'article':
                        priority = 7
                    else:
                        priority = 5
            
                    content_containers.append((element, len(text), priority))

            # Sort by priority first, then content length
            content_containers.sort(key=lambda x: (x[2], x[1]), reverse=True)

            if content_containers:
                main_element = content_containers[0][0]
                print(f"Selected main content container: <{main_element.name}> with {content_containers[0][1]} chars (priority {content_containers[0][2]})")
            else:
                main_element = soup.body
            print("Using <body> as fallback content container")
            
            if main_element:
                # First, remove unwanted elements from the main container
                for unwanted in main_element.select("footer, .footer, .site-footer, #footer, nav.breadcrumb, .breadcrumb, [class*='breadcrumb'], header, .header, .site-header, #header, .site-branding, .global-header, .top-bar, .primary-navigation, .global-navigation, .utility-nav, .top-nav, .main-menu, .cookie-banner, [class*='cookie'], .social-media, .social-links, .share-buttons, [class*='share'], [class*='site-social-bar']"):
                    unwanted.decompose()
    
                # Now extract the content in its natural order
                content_text = main_element.get_text(separator='\n', strip=True)
    
                # Check if we got substantial content
                if len(content_text) < 100:
                    print(f"WARNING: Extracted content is too short ({len(content_text)} chars)")
        
                    # Debug what the page actually contains
                    all_text = soup.body.get_text(separator='\n', strip=True)
                    print(f"Body text length: {len(all_text)} chars")
                    print(f"First 200 chars of body: {all_text[:200]}...")
        
                    # Get text directly from browser
                    browser_text = await page.evaluate("""() => document.body.innerText""")
                    print(f"Browser text length: {len(browser_text)}")
                    print(f"First 200 chars from browser: {browser_text[:200]}...")
            else:
                # Fallback to body content
                body = soup.body
                if body:
                    # First, remove unwanted elements from body
                    for unwanted in body.select("footer, .footer, .site-footer, #footer, nav.breadcrumb, .breadcrumb, [class*='breadcrumb'], header, .header, .site-header, #header, .site-branding, .global-header, .top-bar, nav, .nav, .navigation, .navbar, .menu, .site-navigation, .cookie-banner, [class*='cookie'], .social-media, .social-links, .share-buttons, [class*='share'], [class*='site-social-bar']"):
                        unwanted.decompose()
            
                    content_text = body.get_text(separator='\n', strip=True)
                else:
                    content_text = "No content could be extracted from this page."
            
            # Handle folder paths differently based on direct structure flag
            if self.use_direct_folder_structure:
                # Use the output folder directly without creating nested level_X folders
                level_folder = self.output_folder
            else:
                # Original behavior: create nested level_X folders
                level_folder = os.path.join(self.output_folder, f"level_{depth}")
            
            # Save the content
            safe_filename = self.get_safe_filename(url)
            content_path = os.path.join(level_folder, "text_content", f"{safe_filename}.txt")
            
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(f"URL: {url}\n")
                f.write(f"Base URL: {base_url}\n")
                if fragment:
                    f.write(f"Fragment: #{fragment}\n")
                f.write(f"Site Type: {site_type}\n")
                f.write(f"Crawl Depth: {depth}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 80 + "\n\n")
                f.write(content_text)
            
            # IMPROVED LINK EXTRACTION
            # Create a clean copy of the soup without headers and footers
            clean_soup = BeautifulSoup(str(soup), 'html.parser')
            
            # Explicitly remove header and footer elements from the clean soup
            for header_selector in self.content_selectors["header_elements"].split(','):
                for header in clean_soup.select(header_selector.strip()):
                    header.decompose()
                    
            for footer_selector in self.content_selectors["footer_elements"].split(','):
                for footer in clean_soup.select(footer_selector.strip()):
                    footer.decompose()
            
            # Also remove navigation elements
            for nav_selector in [
                # Only remove GLOBAL navigation elements
                '.primary-navigation', '.global-navigation', '.utility-nav', '.top-nav', '.main-menu',
                '.site-header-navigation', '.main-header-navigation', 'nav:not([class*="navigation-list"])', 
                '.navigation:not([class*="navigation-list"])',
    
                # Keep breadcrumb selectors
                '[class*="breadcrumb"]', '.Breadcrumb', 'ol.Breadcrumb', '.breadcrumb-mobile',
                '[aria-label="Breadcrumb"]', '.breadcrumb-trail', '.crumbs',
    
                # Keep social media selectors
                '.site-social-bar', 'section.site-social-bar', '[class*="social-bar"]',
                '.social-media', '.social-links', '.share-buttons', '[class*="site-social-bar"]'
            ]:
                for nav in clean_soup.select(nav_selector):
                    nav.decompose()
            
            # Initialize link collections
            all_links = []
            internal_links = []
            external_links = []
            fragment_links = []
            
            # Method 1: Extract links directly from HTML using BeautifulSoup - using cleaned soup
            for a_tag in clean_soup.find_all('a', href=True):
                href = a_tag['href']
                if href and not href.startswith('javascript:'):
                    # Process the link - headers and footers already removed
                    full_url = urljoin(base_url, href)  # Use base_url to resolve relative URLs correctly
                    if full_url.startswith(('http://', 'https://')):
                        # Skip links to the same page (common in breadcrumbs)
                        if full_url == url or full_url == base_url:
                            continue

                        # Skip links to parent paths if they're few segments
                        # (This helps avoid breadcrumb parent links)
                        parsed_current = urlparse(url)
                        parsed_link = urlparse(full_url)
                        if (parsed_current.netloc == parsed_link.netloc and 
                            parsed_current.path.startswith(parsed_link.path) and
                            parsed_link.path.count('/') <= 2):
                            continue

                        all_links.append(full_url)
                        
                        # Categorize links as internal or external
                        if self.is_internal_url(full_url):
                            internal_links.append(full_url)
                        else:
                            external_links.append(full_url)
                        
                        # Check if it's a fragment link
                        if '#' in full_url:
                            fragment_links.append(full_url)
                        
            # Method 2: Also use Playwright's selector for backup - with filtering
            try:
                # First remove ONLY global elements before extracting links - MODIFIED
                await page.evaluate("""() => {
                    // Helper function to remove elements by selector
                    function removeElements(selector) {
                        document.querySelectorAll(selector).forEach(el => {
                            if (el) el.remove();
                        });
                    }
                    
                    // Remove ONLY global headers and elements that definitely aren't content
                    removeElements('header:not(.page-header), .site-header, #site-header, .global-header, .site-branding');
                    
                    // Remove footers
                    removeElements('footer, .footer, .site-footer, #footer');
                    
                    // Only remove global navigation, not content navigation
                    removeElements('.global-navigation, .utility-nav, .top-nav');
                                    
                    // Remove social media bars and sharing elements
                    removeElements('.site-social-bar, section.site-social-bar, [class*="social-bar"], .social-media, .social-links, .share-buttons, [class*="site-social-bar"]');
                    
                    // DO NOT remove all navigation - some navigation contains important service links
                }""")
                
                # Get all link elements - after headers and footers are removed
                href_elements = await page.query_selector_all('a[href]')
                for element in href_elements:
                    try:
                        href = await element.get_attribute('href')
                        if href and not href.startswith('javascript:'):
                            full_url = urljoin(base_url, href)  # Use base_url here too
            
                            # Skip links to the same page (common in breadcrumbs)
                            if full_url == url or full_url == base_url:
                                continue
                
                            # Skip links to parent paths if they're few segments
                            parsed_current = urlparse(url)
                            parsed_link = urlparse(full_url)
                            if (parsed_current.netloc == parsed_link.netloc and 
                                parsed_current.path.startswith(parsed_link.path) and
                                parsed_link.path.count('/') <= 2):
                                continue
                
                            if full_url.startswith(('http://', 'https://')) and full_url not in all_links:
                                all_links.append(full_url)
                                
                                # Categorize links as internal or external
                                if self.is_internal_url(full_url):
                                    if full_url not in internal_links:
                                        internal_links.append(full_url)
                                else:
                                    if full_url not in external_links:
                                        external_links.append(full_url)
                                
                                # Check if it's a fragment link
                                if '#' in full_url and full_url not in fragment_links:
                                    fragment_links.append(full_url)
                    except Exception as e:
                        print(f"Error processing link element: {str(e)}")
                        continue

                # NEW CODE: Specifically look for service links, cards, and tiles - common in government sites
                try:
                    # Look for links in card/tile patterns and service containers
                    service_link_selectors = [
                        '.card a', '.tile a', '[class*="card"] a', '[class*="tile"] a',
                        '.widget a', '.widget-row a', '.widget-content a', '.widget-navigation a',
                        '.service a', '[class*="service"] a', '[class*="feature"] a', 
                        '.grid a', '[class*="grid"] a',
                        'a.btn', 'a.button', 'a[role="button"]',

                        # Add these general content selectors that should work across sites
                        'article a',                # Links in article elements
                        'main a',                   # Links in main content 
                        '#content a',               # Links in #content areas
                        '.content a',               # Links in .content areas
                        '[role="main"] a',          # Links in main content areas
    
                        # Common layout system selectors
                        '.row a',                   # Common in Bootstrap/Foundation-based layouts
                        '[class*="col"] a',         # Column patterns in grid systems
    
                        # Common heading and list patterns
                        'h2 a, h3 a, h4 a',         # Links in headings (very common pattern)
                        'ul li a',                  # Links in list items
                        '.navigation-lists-block a', # SPECIFICALLY target the navigation-lists-block
                        '[class*="navigation-list"] a', # Broader pattern to catch similar structures
    
                        # Common content section patterns
                        '[class*="list"] a',        # Lists of links (won't match global nav lists)
                        '[class*="section"] a',     # Section content areas
                        '[class*="block"] a'        # Content blocks
                    ]
                    
                    for selector in service_link_selectors:
                        service_elements = await page.query_selector_all(selector)
                        for element in service_elements:
                            try:
                                href = await element.get_attribute('href')
                                if href and not href.startswith('javascript:'):
                                    full_url = urljoin(base_url, href)  # Use base_url for proper URL resolution
            
                                    # Skip links to the same page (common in breadcrumbs)
                                    if full_url == url or full_url == base_url:
                                        continue
                
                                    # Skip links to parent paths if they're few segments
                                    parsed_current = urlparse(url)
                                    parsed_link = urlparse(full_url)
                                    if (parsed_current.netloc == parsed_link.netloc and 
                                        parsed_current.path.startswith(parsed_link.path) and
                                        parsed_link.path.count('/') <= 2):
                                        continue
                
                                    if full_url.startswith(('http://', 'https://')) and full_url not in all_links:
                                        all_links.append(full_url)
                                        
                                        # Categorize links as internal or external
                                        if self.is_internal_url(full_url):
                                            if full_url not in internal_links:
                                                internal_links.append(full_url)
                                                print(f"Found service link: {full_url}")
                                        else:
                                            if full_url not in external_links:
                                                external_links.append(full_url)
                            except Exception as e:
                                print(f"Error processing service link: {str(e)}")
                                continue
                except Exception as e:
                    print(f"Error finding service links: {str(e)}")
                    print("Continuing with already found links")
                    
            except Exception as e:
                print(f"Error querying link elements: {str(e)}")
                print("Continuing with links found through BeautifulSoup only")

            
            # Debug link collections
            # Uncomment these lines during development to see what links are being found
            # self.debug_links(all_links, "All Links")
            # self.debug_links(internal_links, "Internal Links")
            # self.debug_links(external_links, "External Links") 
            # self.debug_links(fragment_links, "Fragment Links")
            
            # Update stats
            self.stats["links_found"] += len(all_links)
            self.stats["internal_links_found"] += len(internal_links)
            self.stats["fragment_links_count"] += len(fragment_links)
            
            # Debug link extraction
            print(f"Found {len(all_links)} total links")
            print(f"Found {len(internal_links)} internal links")
            print(f"Found {len(external_links)} external links (will be ignored)")
            print(f"Found {len(fragment_links)} fragment links")
                    
            # IMPROVED IMAGE EXTRACTION
            image_links = []
            
            # Method 1: Use BeautifulSoup to find all images
            for img in soup.find_all('img', src=True):
                src = img.get('src')
                if src and not src.startswith('data:'):
                    full_url = urljoin(base_url, src)  # Use base_url here too
                    if full_url.startswith(('http://', 'https://')) and self.is_internal_url(full_url):
                        image_links.append(full_url)
            
            # Method 2: Also use Playwright selectors
            try:
                img_elements = await page.query_selector_all('img[src]')
                for element in img_elements:
                    try:
                        src = await element.get_attribute('src')
                        if src and not src.startswith('data:'):
                            full_url = urljoin(base_url, src)  # Use base_url here too
                            if full_url.startswith(('http://', 'https://')) and self.is_internal_url(full_url) and full_url not in image_links:
                                image_links.append(full_url)
                    except Exception as e:
                        print(f"Error processing image element: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error querying image elements: {str(e)}")
                print("Continuing with images found through BeautifulSoup only")
            
            # Also look for background images in CSS
            try:
                style_elements = await page.query_selector_all('[style*="background-image"]')
                for element in style_elements:
                    try:
                        style = await element.get_attribute('style')
                        if style and 'url(' in style:
                            # Extract URL from background-image: url(...)
                            url_match = re.search(r'url\([\'"]?(.*?)[\'"]?\)', style)
                            if url_match:
                                img_url = url_match.group(1)
                                full_url = urljoin(base_url, img_url)  # Use base_url here too
                                if full_url.startswith(('http://', 'https://')) and self.is_internal_url(full_url) and full_url not in image_links:
                                    image_links.append(full_url)
                    except Exception as e:
                        print(f"Error processing background image: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error querying background images: {str(e)}")
                print("Continuing without background images")
            
            # Find all links containing images (these are often thumbnails linking to full images)
            linked_images = []
            img_link_elements = await page.query_selector_all('a > img, a:has(img)')
            for element in img_link_elements:
                # Get the parent if it's an image
                if await element.evaluate('el => el.tagName === "IMG"'):
                    parent = await element.evaluate('el => el.parentElement.tagName === "A" ? el.parentElement.href : null')
                    if parent:
                        src = await element.get_attribute('src')
                        if src:
                            img_src = urljoin(base_url, src)
                            full_img_url = urljoin(base_url, parent)
                            # Only add internal images
                            if self.is_internal_url(img_src) and self.is_internal_url(full_img_url):
                                linked_images.append((img_src, full_img_url))
                # Get the href and img src if it's an anchor with an image
                elif await element.evaluate('el => el.tagName === "A"'):
                    href = await element.get_attribute('href')
                    img = await element.query_selector('img')
                    if href and img:
                        src = await img.get_attribute('src')
                        if src:
                            img_src = urljoin(base_url, src)
                            full_img_url = urljoin(base_url, href)
                            # Only add internal images
                            if self.is_internal_url(img_src) and self.is_internal_url(full_img_url):
                                linked_images.append((img_src, full_img_url))
            
            # Also look for lightbox images - only internal ones
            lightbox_elements = await page.query_selector_all('[data-lightbox], [data-fancybox], .lightbox, .gallery-item a')
            for element in lightbox_elements:
                href = await element.get_attribute('href')
                if href and any(href.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']):
                    full_img_url = urljoin(base_url, href)
                    # Skip if not internal
                    if not self.is_internal_url(full_img_url):
                        continue
                    # Get the thumbnail if available
                    img = await element.query_selector('img')
                    if img:
                        src = await img.get_attribute('src')
                        img_src = urljoin(base_url, src) if src and self.is_internal_url(urljoin(base_url, src)) else None
                    else:
                        img_src = None
                    linked_images.append((img_src, full_img_url))
            
            # Save base64 images - these are embedded directly in the HTML
            base64_folder = os.path.join(level_folder, "downloaded_files", "embedded_images")
            os.makedirs(base64_folder, exist_ok=True)
            
            for index, (data_url, alt_text) in enumerate(base64_images):
                try:
                    # Parse the data URL
                    if ';base64,' in data_url:
                        header, base64_data = data_url.split(';base64,', 1)
                        mime_type = header.split(':', 1)[1] if ':' in header else 'image/png'
                        
                        # Determine file extension from mime type
                        extension = '.png'  # Default
                        if 'jpeg' in mime_type or 'jpg' in mime_type:
                            extension = '.jpg'
                        elif 'gif' in mime_type:
                            extension = '.gif'
                        elif 'svg' in mime_type:
                            extension = '.svg'
                        elif 'webp' in mime_type:
                            extension = '.webp'
                        
                        # Create safe filename using alt text if available
                        safe_alt = re.sub(r'[\\/*?:"<>|]', '_', alt_text)
                        filename = f"{safe_alt}_{index}{extension}" if safe_alt else f"embedded_image_{index}{extension}"
                        file_path = os.path.join(base64_folder, filename)
                        
                        # Decode and save the image
                        try:
                            image_data = base64.b64decode(base64_data)
                            with open(file_path, 'wb') as f:
                                f.write(image_data)
                            print(f"Saved embedded base64 image: {filename}")
                            self.stats["files_downloaded"] += 1
                        except Exception as e:
                            print(f"Error decoding base64 image: {str(e)}")
                except Exception as e:
                    print(f"Error processing base64 image: {str(e)}")
            
            # Combine image sources
            all_image_links = list(dict.fromkeys(image_links))
            full_size_images = []
            
            # Add linked images (thumbnails and their full-size versions)
            for thumb, full in linked_images:
                if thumb and thumb not in all_image_links:
                    all_image_links.append(thumb)
                if full and full not in all_image_links:
                    all_image_links.append(full)
                    full_size_images.append(full)
            
            # IMPROVED FILE DETECTION
            # Use our improved file detection to find both extension-based files and pattern-based files
            file_links = []
            for link in internal_links:
                if self.is_file_url(link):
                    file_links.append(link)
            
            # Add image links that are media files
            media_links = file_links + [link for link in all_image_links if self.is_file_url(link)]
            
            # Remove duplicates while preserving order
            media_links = list(dict.fromkeys(media_links))
            
            print(f"Found {len(media_links)} file links to download")
            
            # Download files with better tracking and verification
            files_folder = os.path.join(level_folder, "downloaded_files")
            os.makedirs(files_folder, exist_ok=True)
            
            # Track all download attempts and results
            downloaded_files = []
            
            for file_url in media_links:
                try:
                    # Skip if not internal
                    if not self.is_internal_url(file_url):
                        print(f"Skipping external file: {file_url}")
                        downloaded_files.append({
                            "url": file_url,
                            "success": False,
                            "reason": "External URL - skipped",
                            "file_path": None
                        })
                        continue
                    
                    # Skip if already downloaded (using global tracking)
                    if file_url in self.downloaded_files:
                        print(f"Skipping already downloaded file: {file_url}")
                        downloaded_files.append({
                            "url": file_url,
                            "success": True,
                            "reason": "Already downloaded in previous level",
                            "file_path": None
                        })
                        continue
                    
                    # Skip if already attempted but failed (avoid infinite retries)
                    if file_url in self.download_attempts and file_url not in self.downloaded_files:
                        print(f"Skipping previously failed file: {file_url}")
                        downloaded_files.append({
                            "url": file_url,
                            "success": False,
                            "reason": "Previously failed in another level",
                            "file_path": None
                        })
                        continue
                        
                    # Mark this as attempted
                    self.download_attempts.add(file_url)
                    
                    # Get filename
                    filename = os.path.basename(urlparse(file_url).path)
                    if not filename or filename == '' or '.' not in filename:
                        # For URLs without proper filenames like /downloads/file/12345/some-name
                        parts = file_url.split('/')
                        if len(parts) > 3:
                            # Use the last part of the path as the filename
                            filename = parts[-1]
                            # Add extension if needed based on URL pattern
                            if '.' not in filename:
                                # Check specific patterns
                                if '/downloads/file/' in file_url:
                                    # Gov.uk files are often PDFs if not specified
                                    filename += '.pdf'
                                elif '/download/' in file_url:
                                    # Generic download URLs are often PDFs
                                    filename += '.pdf'
                        else:
                            # Generate a timestamp-based filename if can't extract one
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            ext = next((ext for ext in self.file_extensions if file_url.lower().endswith(ext)), '.bin')
                            filename = f"download_{timestamp}{ext}"
                    
                    # Create safe file path
                    file_path = os.path.join(files_folder, self.get_safe_filename(filename))
                    
                    # Check if this file already exists locally
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        print(f"File already exists locally: {file_path}")
                        self.downloaded_files.add(file_url)  # Mark as successfully downloaded
                        
                        # Verify the file
                        is_valid, message = self.verify_downloaded_file(file_path)
                        
                        downloaded_files.append({
                            "url": file_url,
                            "success": True,
                            "reason": f"File already exists locally: {message}",
                            "file_path": file_path,
                            "file_size": os.path.getsize(file_path)
                        })
                        
                        self.stats["files_downloaded"] += 1
                        self.stats["level_stats"][f"level_{depth}"]["files"] += 1
                        continue
                    
                    # Use more visible markers for download attempts
                    print("\n" + "=" * 50)
                    print(f"🔄 DOWNLOAD ATTEMPT: {file_url}")
                    
                    # Special handling for government domains with increased timeouts
                    success = False
                    if self.is_gov_domain and ('/downloads/file/' in file_url or '/documents/' in file_url or '/download/' in file_url):
                        # Use longer timeout for gov domains
                        timeout = self.gov_request_timeout * 1000  # Convert to ms for Playwright
                        
                        # Try downloading with Playwright first for government sites
                        try:
                            print(f"Using Playwright for government file with {timeout}ms timeout")
                            success, need_fallback = await self.download_file_with_playwright(page, file_url, file_path, timeout=timeout, source_page_url=url)
                            
                            if success:
                                # Verify the file
                                is_valid, message = self.verify_downloaded_file(file_path)
                                if is_valid:
                                    # No need for fallback
                                    self.downloaded_files.add(file_url)
                                    downloaded_files.append({
                                        "url": file_url,
                                        "success": True,
                                        "reason": f"Downloaded with Playwright: {message}",
                                        "file_path": file_path,
                                        "file_size": os.path.getsize(file_path)
                                    })
                                    self.stats["files_downloaded"] += 1
                                    self.stats["level_stats"][f"level_{depth}"]["files"] += 1
                                    print(f"✅ SUCCESS: Downloaded {file_url} ({os.path.getsize(file_path)} bytes)")
                                    print("=" * 50 + "\n")
                                    continue
                                else:
                                    # File verification failed
                                    print(f"File verification failed: {message}")
                                    success = False
                                    # Will try fallback
                            
                            elif not need_fallback:
                                # Something went wrong, but don't try fallback
                                downloaded_files.append({
                                    "url": file_url,
                                    "success": False,
                                    "reason": "Playwright download failed (no fallback needed)",
                                    "file_path": None
                                })
                                print(f"❌ FAILED: Could not download {file_url}")
                                print("=" * 50 + "\n")
                                continue
                            # If need_fallback is True, we'll continue to the requests method
                        except Exception as e:
                            print(f"Playwright download attempt failed: {str(e)}")
                            print(f"Falling back to requests method...")
                    
                    # If Playwright method didn't succeed or wasn't used, try with requests
                    if not success:
                        print(f"Using requests for file: {file_url}")
                        
                        # Use longer timeout for gov domains
                        req_timeout = self.gov_request_timeout if self.is_gov_domain else self.request_timeout
                        
                        time.sleep(random.uniform(1.0, 2.0))  # Longer delay for fallback
                        success = self.download_file(file_url, file_path, timeout=req_timeout, source_page_url=url)
                        
                        if success:
                            # Verify the file
                            is_valid, message = self.verify_downloaded_file(file_path)
                            if is_valid:
                                downloaded_files.append({
                                    "url": file_url,
                                    "success": True,
                                    "reason": f"Downloaded with requests: {message}",
                                    "file_path": file_path,
                                    "file_size": os.path.getsize(file_path)
                                })
                                self.stats["files_downloaded"] += 1
                                self.stats["level_stats"][f"level_{depth}"]["files"] += 1
                                print(f"✅ SUCCESS: Downloaded {file_url} ({os.path.getsize(file_path)} bytes)")
                            else:
                                # File verification failed
                                downloaded_files.append({
                                    "url": file_url,
                                    "success": False,
                                    "reason": f"File verification failed: {message}",
                                    "file_path": file_path
                                })
                                print(f"❌ FAILED: Downloaded file verification failed: {message}")
                        else:
                            downloaded_files.append({
                                "url": file_url,
                                "success": False,
                                "reason": "Both download methods failed",
                                "file_path": None
                            })
                            print(f"❌ FAILED: Could not download {file_url}")
                    
                    print("=" * 50 + "\n")
                    
                except Exception as e:
                    print(f"Error processing download for {file_url}: {str(e)}")
                    downloaded_files.append({
                        "url": file_url,
                        "success": False,
                        "reason": f"Exception: {str(e)}",
                        "file_path": None
                    })
                    print(f"❌ FAILED: Exception while downloading {file_url}")
                    print("=" * 50 + "\n")
            
            # Save download summary at the end
            self.save_download_summary(level_folder, media_links, downloaded_files)
            
            # Get internal links for next level - ONLY internal links
            next_level_links = []
            
            # Debug which links are being skipped and why
            print("\nDEBUG: INTERNAL LINKS FILTERING")
            print("---------------------------------")
            
            # Process candidate links
            candidate_links = []
            for i, link in enumerate(internal_links):
                # Check if this URL should be skipped
                if self.should_skip_url(link):
                    # Check which pattern is causing the skip
                    if not link.startswith(('http://', 'https://')):
                        skip_reason = "Not an HTTP/HTTPS URL"
                    else:
                        for pattern in self.exclude_patterns:
                            if re.search(pattern, link):
                                skip_reason = f"Matches exclude pattern: {pattern}"
                                break
                    print(f"SKIP  [{i+1}] {link} - Reason: {skip_reason}")
                    continue
                # Skip file links (images, PDFs, etc.) - they should be downloaded but not crawled
                if self.is_file_url(link):
                    print(f"SKIP  [{i+1}] {link} - Reason: File link (will be downloaded but not crawled)")
                    continue
                
                # Add to candidates for further processing
                candidate_links.append((i+1, link))
            
            # Check which links have already been visited
            for i, link in candidate_links:
                # Check if we've already visited - using normalized URLs for efficiency
                normalized_link = self.normalize_url(link)
                if normalized_link in self.normalized_visited_urls:
                    print(f"SKIP  [{i}] {link} - Reason: Already visited (normalized)")
                    continue
                
                # For URLs with fragments, check if we've visited the base URL
                link_base, link_fragment = urldefrag(link)
                normalized_base = self.normalize_url(link_base)
                if link_fragment and normalized_base in self.normalized_base_urls:
                    print(f"SKIP  [{i}] {link} - Reason: Already visited the base URL (without fragment)")
                    continue
                
                # If we get here, the URL is new and should be visited
                print(f"FOLLOW [{i}] {link}")
                next_level_links.append((link, depth + 1))
            
            print("---------------------------------")
            
            # Create detailed metadata about the extracted links
            link_metadata = {
                "url": url,
                "title": soup.title.string if soup.title else "No title",
                "base_url": base_url,
                "fragment": fragment if fragment else None,
                "site_type": site_type,
                "crawl_depth": depth,
                "all_links_count": len(all_links),
                "internal_links_count": len(internal_links),
                "external_links_count": len(external_links),
                "fragment_links_count": len(fragment_links),
                "followable_links_count": len(next_level_links),
                "file_links_count": len(file_links),
                "image_links_count": len(all_image_links),
                "full_size_images_count": len(full_size_images),
                "base64_images_count": len(base64_images),
                "internal_links": internal_links,
                "external_links": external_links[:50],  # Limit to avoid huge JSON files
                "fragment_links": fragment_links[:30],  # Limit to avoid huge JSON files
                "followable_links": [link for link, _ in next_level_links],
                "file_links": file_links,
                "image_links": all_image_links[:100],  # Limit to avoid huge JSON files
                "full_size_images": full_size_images[:50],  # Full-size versions of thumbnails
                "base64_images": [alt for _, alt in base64_images]  # Just save the alt text
            }
            
            # Add to the collection of all links for evaluation
            self.all_collected_links.append(link_metadata)
            
            # Save links to JSON
            links_path = os.path.join(level_folder, "text_content", f"{safe_filename}_links.json")
            with open(links_path, "w", encoding="utf-8") as f:
                json.dump(link_metadata, f, indent=2)
            
            # Save stats after each page to enable resuming
            stats_path = os.path.join(self.output_folder, "scraping_stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)
            
            # Return links to visit next if not at max depth
            if depth < self.max_depth:
                print(f"Found {len(next_level_links)} new internal links to follow for depth {depth + 1}")
                return next_level_links
            else:
                return []
                
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    async def crawl(self):
        """Simple sequential crawler that avoids event loop issues."""
        # MODIFIED: Create folders based on the direct structure flag
        if self.use_direct_folder_structure:
            # When called from CrawlerCoordinator, use the folders that were already created
            print(f"Using existing folder structure at: {self.output_folder}")
        else:
            # Original behavior for standalone use - create fresh output directory
            if os.path.exists(self.output_folder):
                shutil.rmtree(self.output_folder)
            print(f"Creating fresh output directory: {self.output_folder}")
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Create the level folders
            for level in range(self.max_depth + 1):
                os.makedirs(os.path.join(self.output_folder, f"level_{level}/text_content"), exist_ok=True)
                os.makedirs(os.path.join(self.output_folder, f"level_{level}/downloaded_files"), exist_ok=True)
        
        # Initialize the queue with the starting URLs
        url_queue = [(url, 0) for url in self.seed_urls]
        
        # Record start time
        start_time = datetime.now()
        
        # Start the browser
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--disable-web-security',
                    '--window-size=1920,1080',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--ignore-certificate-errors'
                ]
            )
          
            # Create a browser context with improved settings
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                accept_downloads=True,
                ignore_https_errors=True,
                locale='en-GB',  # UK locale for gov.uk sites
            )

            # Hide automation
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false
                });
            """)
            
            # Create a single page for all processing
            page = await context.new_page()
            
            try:
                # Process URLs one at a time
                while url_queue:
                    url, depth = url_queue.pop(0)  # Get next URL from the queue
                    
                    # Skip if already visited (using efficient normalized check)
                    normalized_url = self.normalize_url(url)
                    if normalized_url in self.normalized_visited_urls:
                        print(f"Already visited (normalized): {url}")
                        continue
                    
                    # Mark as visited before processing
                    self.visited_urls.add(url)
                    base_url, _ = urldefrag(url)
                    self.visited_base_urls.add(base_url)
                    self.normalized_visited_urls.add(normalized_url)
                    self.normalized_base_urls.add(self.normalize_url(base_url))
                    
                    # Process the page
                    new_links = await self.process_page_with_timeout(page, url, depth)
                    
                    # Add new links to the queue
                    for link, new_depth in new_links:
                        # Use normalized URLs for checking
                        normalized_link = self.normalize_url(link)
                        if normalized_link not in self.normalized_visited_urls:
                            url_queue.append((link, new_depth))
                    
            finally:
                await page.close()
                await browser.close()
        
        # Record completion time and duration
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Save final stats
        self.stats["duration"] = str(duration)
        self.stats["start_time"] = start_time.isoformat()
        self.stats["end_time"] = end_time.isoformat()
        
        stats_path = os.path.join(self.output_folder, "scraping_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)
        
        # Save all collected links for evaluation
        links_for_evaluation_path = os.path.join(self.output_folder, "links_for_evaluation.json")
        with open(links_for_evaluation_path, "w", encoding="utf-8") as f:
            json.dump(self.all_collected_links, f, indent=2)
        
        print(f"Scraping completed in {duration}!")
        print(f"Visited {self.stats['pages_visited']} pages")
        print(f"Downloaded {self.stats['files_downloaded']} files")
        print(f"Found {self.stats['links_found']} total links")
        print(f"Found {self.stats['internal_links_found']} internal links")
        print(f"Found {self.stats['fragment_links_count']} fragment links")
        print(f"Results saved to {self.output_folder}")
        print(f"Links for evaluation saved to {links_for_evaluation_path}")

        return self.all_collected_links

def main():
    """Main entry point that properly uses asyncio.run()."""
    parser = argparse.ArgumentParser(description='Web Scraper for Local Authority websites')
    parser.add_argument('--url', type=str, help='Starting URL to crawl', required=True)
    parser.add_argument('--depth', type=int, default=1, help='Maximum crawl depth (default: 1)')
    parser.add_argument('--output', type=str, default="scraper_output", help='Output folder')
    parser.add_argument('--min-delay', type=float, default=1.5, help='Minimum delay between requests')
    parser.add_argument('--max-delay', type=float, default=3.0, help='Maximum delay between requests')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    
    args = parser.parse_args()
    
    # Initialize the scraper with command line arguments
    scraper = LevelScraper(
        seed_urls=[args.url], 
        max_depth=args.depth,
        output_folder=args.output,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        request_timeout=args.timeout
    )
    
    # Use asyncio.run() to create and manage the event loop
    asyncio.run(scraper.crawl())

if __name__ == "__main__":
    main()