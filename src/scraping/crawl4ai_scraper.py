"""Enhanced web content scraping using Crawl4AI."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import ScrapedContent, WebSearchResult
from config.settings import settings

# Setup logging
logger = logging.getLogger(__name__)

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter
    CRAWL4AI_AVAILABLE = True
except ImportError:
    logger.warning("crawl4ai not available. Please install with: pip install crawl4ai")
    CRAWL4AI_AVAILABLE = False


class Crawl4AIScraper:
    """Enhanced web content scraper using Crawl4AI."""

    def __init__(self,
                 headless: bool = True,
                 request_delay: float = 1.0,
                 timeout: int = 30,
                 max_concurrent: int = 5,
                 use_stealth: bool = True):
        """Initialize the Crawl4AI scraper."""
        if not CRAWL4AI_AVAILABLE:
            raise ImportError("crawl4ai is required but not installed. Run: pip install crawl4ai")

        self.request_delay = request_delay
        self.timeout = timeout
        self.max_concurrent = max_concurrent

        # Configure browser for better success rates
        self.browser_config = BrowserConfig(
            headless=headless,
            browser_type="chromium",
            user_agent_mode="random",  # Random user agents for better stealth
            verbose=False,
            ignore_https_errors=True,
            java_script_enabled=True,
            accept_downloads=False,
            extra_args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-gpu",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-features=TranslateUI",
                "--disable-ipc-flooding-protection",
            ] if use_stealth else None
        )

        # Configure crawler for optimal content extraction with fallback approach
        try:
            # Try to configure with content filter (newer versions)
            content_filter = PruningContentFilter(
                threshold=0.48,  # Keep more content than default
                threshold_type="fixed",
                min_word_threshold=20
            )

            markdown_generator = DefaultMarkdownGenerator(
                content_filter=content_filter
            )

            self.crawler_config = CrawlerRunConfig(
                cache_mode=CacheMode.ENABLED,
                word_count_threshold=50,  # Minimum words for valid content
                extraction_strategy=None,  # Will be set per request if needed
                markdown_generator=markdown_generator,
                scan_full_page=True,  # Handle lazy loading
                process_iframes=False,  # Skip iframes for performance
                remove_overlay_elements=True,  # Remove popups/modals
                delay_before_return_html=1.0,  # Wait for dynamic content
            )
            logger.info("Crawl4AI configured with content filtering")
        except Exception as e:
            # Fallback to basic configuration for older versions
            logger.warning(f"Advanced Crawl4AI configuration failed, using basic config: {e}")
            self.crawler_config = CrawlerRunConfig(
                cache_mode=CacheMode.ENABLED,
                word_count_threshold=50,
                scan_full_page=True,
                process_iframes=False,
                remove_overlay_elements=True,
                delay_before_return_html=1.0,
            )

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file."""
        return url.lower().endswith('.pdf')

    async def scrape_content(self, url: str, extract_schema: Optional[Dict] = None) -> ScrapedContent:
        """
        Scrape content from a single URL using Crawl4AI.
        Supports both web pages and PDF files.

        Args:
            url: The URL to scrape (web page or PDF)
            extract_schema: Optional JSON-CSS extraction schema

        Returns:
            ScrapedContent object with scraped data
        """
        try:
            logger.info(f"Starting Crawl4AI scraping for {url}")
            is_pdf = self._is_pdf_url(url)

            if is_pdf:
                logger.info(f"PDF detected: {url} - using PDF parsing configuration")

            # Configure extraction strategy if schema provided
            if extract_schema and not is_pdf:  # CSS extraction doesn't work on PDFs
                try:
                    # Try advanced configuration with extraction strategy
                    content_filter = PruningContentFilter(
                        threshold=0.48,
                        threshold_type="fixed",
                        min_word_threshold=20
                    )
                    markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)

                    config = CrawlerRunConfig(
                        cache_mode=CacheMode.ENABLED,
                        word_count_threshold=50,
                        markdown_generator=markdown_generator,
                        extraction_strategy=JsonCssExtractionStrategy(
                            schema=extract_schema,
                            verbose=True
                        ),
                        scan_full_page=True,
                        process_iframes=False,
                        remove_overlay_elements=True,
                        delay_before_return_html=1.0,
                    )
                except Exception as e:
                    # Fallback to basic config with extraction strategy only
                    logger.warning(f"Advanced config with extraction failed, using basic: {e}")
                    config = CrawlerRunConfig(
                        cache_mode=CacheMode.ENABLED,
                        word_count_threshold=50,
                        extraction_strategy=JsonCssExtractionStrategy(
                            schema=extract_schema,
                            verbose=True
                        ),
                        scan_full_page=True,
                        process_iframes=False,
                        remove_overlay_elements=True,
                        delay_before_return_html=1.0,
                    )
            elif is_pdf:
                # Special configuration for PDF files
                try:
                    # Try advanced configuration for PDFs
                    content_filter = PruningContentFilter(
                        threshold=0.3,  # Lower threshold for PDFs (often have shorter paragraphs)
                        threshold_type="fixed",
                        min_word_threshold=10
                    )
                    markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)

                    config = CrawlerRunConfig(
                        cache_mode=CacheMode.ENABLED,
                        word_count_threshold=20,  # Lower threshold for PDFs
                        markdown_generator=markdown_generator,
                        scan_full_page=False,  # PDFs don't need scrolling
                        process_iframes=False,
                        remove_overlay_elements=False,  # PDFs don't have overlays
                        delay_before_return_html=0.5,  # Less delay needed for PDFs
                        wait_until="networkidle",  # Wait for PDF to load
                        page_timeout=60000,  # PDFs might take longer to process
                    )
                except Exception as e:
                    # Fallback to basic config for PDFs
                    logger.warning(f"Advanced PDF config failed, using basic: {e}")
                    config = CrawlerRunConfig(
                        cache_mode=CacheMode.ENABLED,
                        word_count_threshold=20,
                        scan_full_page=False,
                        process_iframes=False,
                        remove_overlay_elements=False,
                        delay_before_return_html=0.5,
                        wait_until="networkidle",
                        page_timeout=60000,
                    )
            else:
                config = self.crawler_config

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=config
                )

                if not result.success:
                    error_message = getattr(result, 'error_message', None) or "Unknown crawl4ai error"
                    return ScrapedContent(
                        url=url,
                        title="Scraping Failed",
                        content="",
                        content_length=0,
                        success=False,
                        error_message=error_message
                    )

                # Extract content - prefer fit_markdown for better quality
                content = result.markdown.fit_markdown if result.markdown else ""
                if not content and result.markdown:
                    content = result.markdown.raw_markdown

                # For PDFs, check if content extraction was successful
                if is_pdf and len(content.strip()) < 50:
                    logger.warning(f"PDF content extraction yielded minimal text for {url} - may be image-heavy or presentation format")
                    # Try to get raw HTML content as fallback
                    if hasattr(result, 'html') and result.html:
                        content = f"PDF Content (raw): {result.html[:1000]}"  # Limit to avoid huge content
                    elif len(content.strip()) == 0:
                        content = f"PDF file detected but minimal text content extracted. This may be an image-heavy document, presentation, or scanned PDF: {url}"

                # Get title from metadata or result with proper validation
                title = "No title found"
                if result.metadata and 'title' in result.metadata and result.metadata['title']:
                    title = str(result.metadata['title']).strip()
                elif hasattr(result, 'title') and result.title:
                    title = str(result.title).strip()

                # Ensure title is never None or empty
                if not title or title.strip() == '':
                    if is_pdf:
                        title = f"PDF Document: {url.split('/')[-1]}"
                    else:
                        title = "No title found"

                # Prepare metadata with safe attribute access
                metadata = {
                    'method': 'crawl4ai',
                    'content_type': 'pdf' if is_pdf else 'webpage',
                    'status_code': getattr(result, 'status_code', None),
                    'timestamp': getattr(result, 'timestamp', None),
                }

                # Convert timestamp to ISO format if it exists and is a datetime object
                if metadata['timestamp']:
                    try:
                        if hasattr(metadata['timestamp'], 'isoformat'):
                            metadata['timestamp'] = metadata['timestamp'].isoformat()
                        else:
                            metadata['timestamp'] = str(metadata['timestamp'])
                    except Exception:
                        # If timestamp conversion fails, just use current time
                        from datetime import datetime
                        metadata['timestamp'] = datetime.now().isoformat()

                # Add extracted structured data if available
                extracted_content = getattr(result, 'extracted_content', None)
                if extracted_content:
                    metadata['extracted_data'] = extracted_content

                # Add media information if available
                media = getattr(result, 'media', None)
                if media:
                    metadata['media'] = media

                # Add links if available
                links = getattr(result, 'links', None)
                if links:
                    metadata['links'] = links

                # Determine success based on content type
                content_length = len(content.strip())

                # For PDFs, be more lenient with content length requirements
                if is_pdf:
                    # PDFs are successful if we got any content or if we at least detected it's a PDF
                    success = content_length > 0
                    if not success:
                        logger.warning(f"PDF extraction failed - no content extracted from {url}")
                else:
                    # For web pages, use standard content length validation
                    success = content_length >= 50

                return ScrapedContent(
                    url=url,
                    title=title,
                    content=content,
                    content_length=content_length,
                    success=success,
                    metadata=metadata
                )

        except Exception as e:
            logger.error(f"Crawl4AI scraping failed for {url}: {str(e)}")

            # Provide more specific error messages for PDFs
            is_pdf = self._is_pdf_url(url)
            if is_pdf:
                error_msg = f"PDF scraping error: {str(e)}. This may be an image-heavy presentation or scanned document."
                title = f"PDF Scraping Failed: {url.split('/')[-1]}"
            else:
                error_msg = f"Crawl4AI error: {str(e)}"
                title = "Scraping Failed"

            return ScrapedContent(
                url=url,
                title=title,
                content="",
                content_length=0,
                success=False,
                error_message=error_msg,
                metadata={'method': 'crawl4ai', 'content_type': 'pdf' if is_pdf else 'webpage'}
            )

    async def scrape_multiple_sources(self,
                                    sources: List[WebSearchResult],
                                    extract_schema: Optional[Dict] = None) -> List[ScrapedContent]:
        """
        Scrape content from multiple sources concurrently.

        Args:
            sources: List of WebSearchResult objects to scrape
            extract_schema: Optional JSON-CSS extraction schema

        Returns:
            List of ScrapedContent objects
        """
        logger.info(f"Starting concurrent Crawl4AI scraping for {len(sources)} sources")

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def scrape_with_semaphore(source: WebSearchResult) -> ScrapedContent:
            async with semaphore:
                # Add delay between requests for politeness
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
                return await self.scrape_content(source.url, extract_schema)

        # Run all scraping tasks concurrently
        tasks = [scrape_with_semaphore(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        scraped_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception scraping {sources[i].url}: {str(result)}")
                scraped_results.append(ScrapedContent(
                    url=sources[i].url,
                    title="Scraping Failed",
                    content="",
                    content_length=0,
                    success=False,
                    error_message=f"Exception: {str(result)}"
                ))
            else:
                scraped_results.append(result)

        success_count = sum(1 for r in scraped_results if r.success)
        logger.info(f"Crawl4AI scraping completed. Success rate: {success_count}/{len(sources)}")

        return scraped_results

    async def scrape_with_extraction(self,
                                   url: str,
                                   css_schema: Dict) -> ScrapedContent:
        """
        Scrape content with structured data extraction using CSS selectors.
        Note: CSS extraction is not available for PDF files.

        Args:
            url: The URL to scrape
            css_schema: JSON-CSS schema for data extraction

        Returns:
            ScrapedContent with extracted structured data
        """
        if self._is_pdf_url(url):
            logger.warning(f"CSS extraction not supported for PDF: {url}")
            return await self.scrape_content(url)  # Scrape without schema
        return await self.scrape_content(url, css_schema)

    async def scrape_pdf(self, url: str) -> ScrapedContent:
        """
        Scrape content from a PDF file using Crawl4AI.

        Args:
            url: The URL of the PDF file to scrape

        Returns:
            ScrapedContent object with PDF text content
        """
        if not self._is_pdf_url(url):
            logger.warning(f"URL does not appear to be a PDF: {url}")

        logger.info(f"Scraping PDF: {url}")
        return await self.scrape_content(url)

    async def scrape_dynamic_content(self,
                                   url: str,
                                   wait_for_selector: Optional[str] = None,
                                   scroll_to_bottom: bool = False,
                                   click_selectors: Optional[List[str]] = None) -> ScrapedContent:
        """
        Scrape dynamic content that requires interaction.

        Args:
            url: The URL to scrape
            wait_for_selector: CSS selector to wait for before extracting
            scroll_to_bottom: Whether to scroll to bottom to trigger lazy loading
            click_selectors: CSS selectors to click (e.g., "Load More" buttons)

        Returns:
            ScrapedContent object
        """
        try:
            # Build JavaScript code for interactions
            js_code = []

            if scroll_to_bottom:
                js_code.append("""
                    // Scroll to bottom to trigger lazy loading
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(resolve => setTimeout(resolve, 2000));
                """)

            if click_selectors:
                for selector in click_selectors:
                    js_code.append(f"""
                        // Click on {selector}
                        const element = document.querySelector('{selector}');
                        if (element) {{
                            element.click();
                            await new Promise(resolve => setTimeout(resolve, 1000));
                        }}
                    """)

            # Configure for dynamic content with fallback
            try:
                # Try advanced configuration
                content_filter = PruningContentFilter(
                    threshold=0.48,
                    threshold_type="fixed",
                    min_word_threshold=20
                )
                markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)

                config = CrawlerRunConfig(
                    cache_mode=CacheMode.ENABLED,
                    word_count_threshold=50,
                    markdown_generator=markdown_generator,
                    scan_full_page=True,
                    process_iframes=False,
                    remove_overlay_elements=True,
                    wait_for_selector=wait_for_selector,
                    js_code=js_code if js_code else None,
                    delay_before_return_html=3.0,  # More time for dynamic content
                )
            except Exception as e:
                # Fallback to basic configuration
                logger.warning(f"Advanced dynamic config failed, using basic: {e}")
                config = CrawlerRunConfig(
                    cache_mode=CacheMode.ENABLED,
                    word_count_threshold=50,
                    scan_full_page=True,
                    process_iframes=False,
                    remove_overlay_elements=True,
                    wait_for_selector=wait_for_selector,
                    js_code=js_code if js_code else None,
                    delay_before_return_html=3.0,
                )

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(url=url, config=config)

                if not result.success:
                    error_message = getattr(result, 'error_message', None) or "Dynamic content scraping failed"
                    return ScrapedContent(
                        url=url,
                        title="Scraping Failed",
                        content="",
                        content_length=0,
                        success=False,
                        error_message=error_message
                    )

                content = result.markdown.fit_markdown if result.markdown else ""
                title = result.metadata.get('title', 'No title found') if result.metadata else 'No title found'

                return ScrapedContent(
                    url=url,
                    title=title,
                    content=content,
                    content_length=len(content),
                    success=True,
                    metadata={
                        'method': 'crawl4ai_dynamic',
                        'status_code': getattr(result, 'status_code', None),
                        'interactions_performed': bool(js_code),
                        'wait_selector': wait_for_selector,
                    }
                )

        except Exception as e:
            logger.error(f"Dynamic content scraping failed for {url}: {str(e)}")
            return ScrapedContent(
                url=url,
                title="Scraping Failed",
                content="",
                content_length=0,
                success=False,
                error_message=f"Dynamic scraping error: {str(e)}"
            )


# Global scraper instance
_crawl4ai_scraper = None

def get_crawl4ai_scraper() -> Crawl4AIScraper:
    """Get a global Crawl4AI scraper instance."""
    global _crawl4ai_scraper
    if _crawl4ai_scraper is None:
        # Use settings if available
        delay = getattr(settings, 'SCRAPING_DELAY', 1.0)
        timeout = getattr(settings, 'SCRAPING_TIMEOUT', 30)
        max_concurrent = getattr(settings, 'MAX_CONCURRENT_SCRAPES', 5)
        _crawl4ai_scraper = Crawl4AIScraper(
            request_delay=delay,
            timeout=timeout,
            max_concurrent=max_concurrent
        )
    return _crawl4ai_scraper


async def scrape_content_crawl4ai(url: str, extract_schema: Optional[Dict] = None) -> ScrapedContent:
    """
    Scrape content from a single URL using Crawl4AI.

    Args:
        url: The URL to scrape
        extract_schema: Optional JSON-CSS extraction schema

    Returns:
        ScrapedContent object
    """
    scraper = get_crawl4ai_scraper()
    return await scraper.scrape_content(url, extract_schema)


async def scrape_multiple_sources_crawl4ai(sources: List[WebSearchResult],
                                         extract_schema: Optional[Dict] = None) -> List[ScrapedContent]:
    """
    Scrape content from multiple sources using Crawl4AI.

    Args:
        sources: List of WebSearchResult objects to scrape
        extract_schema: Optional JSON-CSS extraction schema

    Returns:
        List of ScrapedContent objects
    """
    scraper = get_crawl4ai_scraper()
    return await scraper.scrape_multiple_sources(sources, extract_schema)
