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

    async def scrape_content(self, url: str, extract_schema: Optional[Dict] = None) -> ScrapedContent:
        """
        Scrape content from a single URL using Crawl4AI.

        Args:
            url: The URL to scrape
            extract_schema: Optional JSON-CSS extraction schema

        Returns:
            ScrapedContent object with scraped data
        """
        try:
            logger.info(f"Starting Crawl4AI scraping for {url}")

            # Configure extraction strategy if schema provided
            if extract_schema:
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

                # Get title from metadata or result
                title = "No title found"
                if result.metadata and 'title' in result.metadata:
                    title = result.metadata['title']
                elif hasattr(result, 'title') and result.title:
                    title = result.title

                # Prepare metadata with safe attribute access
                metadata = {
                    'method': 'crawl4ai',
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

                return ScrapedContent(
                    url=url,
                    title=title,
                    content=content,
                    content_length=len(content),
                    success=True,
                    metadata=metadata
                )

        except Exception as e:
            logger.error(f"Crawl4AI scraping failed for {url}: {str(e)}")
            return ScrapedContent(
                url=url,
                title="Scraping Failed",
                content="",
                content_length=0,
                success=False,
                error_message=f"Crawl4AI error: {str(e)}"
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

        Args:
            url: The URL to scrape
            css_schema: JSON-CSS schema for data extraction

        Returns:
            ScrapedContent with extracted structured data
        """
        return await self.scrape_content(url, css_schema)

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
