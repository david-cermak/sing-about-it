"""Web content scraping functionality with Crawl4AI integration."""

import time
import asyncio
import aiohttp
from typing import List, Optional
import sys
from pathlib import Path
import requests
from urllib.parse import urlparse
from fake_useragent import UserAgent
import logging

# Web scraping libraries
from newspaper import Article
from bs4 import BeautifulSoup
from readability import Document

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import ScrapedContent, WebSearchResult
from config.settings import settings

# Try to import Crawl4AI scraper
try:
    from scraping.crawl4ai_scraper import get_crawl4ai_scraper, CRAWL4AI_AVAILABLE
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.warning("Crawl4AI scraper not available, using fallback methods only")

# Setup logging
logger = logging.getLogger(__name__)

# User agent for respectful scraping
ua = UserAgent()


class WebScraper:
    """Web content scraper with multiple extraction methods."""

    def __init__(self, request_delay: float = 2.0, timeout: int = 30):
        """Initialize scraper with configuration."""
        self.request_delay = request_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and scrapeable."""
        try:
            parsed = urlparse(url)
            if not (bool(parsed.netloc) and parsed.scheme in ['http', 'https']):
                return False

            # PDF files are now supported via Crawl4AI
            if url.lower().endswith('.pdf'):
                logger.info(f"PDF file detected: {url} - will use Crawl4AI for PDF extraction")

            return True
        except Exception:
            return False

    def _extract_with_newspaper(self, url: str) -> Optional[ScrapedContent]:
        """Extract content using newspaper3k library."""
        try:
            logger.info(f"Trying newspaper3k extraction for {url}")

            article = Article(url)
            article.config.request_timeout = self.timeout
            article.config.browser_user_agent = ua.random

            article.download()
            article.parse()

            # Basic validation
            if not article.text or len(article.text.strip()) < 100:
                logger.warning(f"Newspaper3k extracted too little content from {url}")
                return None

            return ScrapedContent(
                url=url,
                title=article.title or "No title found",
                content=article.text,
                content_length=len(article.text),
                success=True,
                metadata={
                    'method': 'newspaper3k',
                    'authors': article.authors,
                    'publish_date': str(article.publish_date) if article.publish_date else None,
                    'top_image': article.top_image,
                    'meta_keywords': article.meta_keywords
                }
            )

        except Exception as e:
            logger.warning(f"Newspaper3k failed for {url}: {str(e)}")
            return None

    def _extract_with_readability(self, url: str) -> Optional[ScrapedContent]:
        """Extract content using readability-lxml."""
        try:
            logger.info(f"Trying readability extraction for {url}")

            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            doc = Document(response.content)

            # Extract clean content
            content = doc.summary()
            title = doc.title()

            # Convert HTML to text
            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)

            # Basic validation
            if not text_content or len(text_content.strip()) < 100:
                logger.warning(f"Readability extracted too little content from {url}")
                return None

            return ScrapedContent(
                url=url,
                title=title or "No title found",
                content=text_content,
                content_length=len(text_content),
                success=True,
                metadata={
                    'method': 'readability',
                    'response_status': response.status_code,
                    'content_type': response.headers.get('content-type', 'unknown')
                }
            )

        except Exception as e:
            logger.warning(f"Readability failed for {url}: {str(e)}")
            return None

    def _extract_with_beautifulsoup(self, url: str) -> Optional[ScrapedContent]:
        """Fallback extraction using BeautifulSoup with basic heuristics."""
        try:
            logger.info(f"Trying BeautifulSoup extraction for {url}")

            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer',
                               'aside', 'advertisement', 'ads']):
                element.decompose()

            # Try to find title
            title = None
            title_selectors = ['h1', 'title', '.title', '#title', '.article-title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break

            # Try to find main content
            content_selectors = [
                'article', '.article', '#article',
                '.content', '#content', '.post-content',
                '.entry-content', '.article-content',
                'main', '[role="main"]'
            ]

            content_elem = None
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    break

            # If no specific content area found, use body
            if not content_elem:
                content_elem = soup.find('body')

            if not content_elem:
                logger.warning(f"No content found with BeautifulSoup for {url}")
                return None

            # Extract text content
            text_content = content_elem.get_text(separator=' ', strip=True)

            # Basic validation
            if not text_content or len(text_content.strip()) < 100:
                logger.warning(f"BeautifulSoup extracted too little content from {url}")
                return None

            return ScrapedContent(
                url=url,
                title=title or "No title found",
                content=text_content,
                content_length=len(text_content),
                success=True,
                metadata={
                    'method': 'beautifulsoup',
                    'response_status': response.status_code,
                    'content_type': response.headers.get('content-type', 'unknown')
                }
            )

        except Exception as e:
            logger.warning(f"BeautifulSoup failed for {url}: {str(e)}")
            return None

    async def _try_crawl4ai(self, url: str) -> Optional[ScrapedContent]:
        """Try to scrape using Crawl4AI first."""
        if not CRAWL4AI_AVAILABLE:
            return None

        try:
            logger.info(f"Trying Crawl4AI extraction for {url}")
            scraper = get_crawl4ai_scraper()
            result = await scraper.scrape_content(url)

            # Check if it's a PDF - use different validation criteria
            is_pdf = url.lower().endswith('.pdf')

            if is_pdf:
                # For PDFs, any result is better than traditional methods (which can't handle PDFs)
                if result:
                    logger.info(f"Crawl4AI processed PDF {url} - success: {result.success}, content: {result.content_length} chars")
                    return result  # Return PDF results even if minimal content
                else:
                    logger.warning(f"Crawl4AI failed to process PDF {url}")
                    return None
            else:
                # For web pages, use original validation
                if result and result.success and result.content_length > 100:
                    logger.info(f"Crawl4AI successfully scraped {url}")
                    return result
                else:
                    logger.warning(f"Crawl4AI extraction insufficient for {url}")
                    return None

        except Exception as e:
            logger.warning(f"Crawl4AI failed for {url}: {str(e)}")
            return None

    def scrape_content(self, url: str) -> ScrapedContent:
        """
        Scrape content from a single URL using multiple extraction methods.

        Tries Crawl4AI first for best results, then falls back to newspaper3k,
        readability, and BeautifulSoup.
        """
        if not self._is_valid_url(url):
            return ScrapedContent(
                url=url,
                title="Invalid URL",
                content="",
                content_length=0,
                success=False,
                error_message="Invalid URL format or unsupported content type (e.g., PDF)"
            )

        logger.info(f"Starting scraping for {url}")

        # Try Crawl4AI first if available
        if CRAWL4AI_AVAILABLE:
            try:
                # Run async crawl4ai in sync context
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                crawl4ai_result = loop.run_until_complete(self._try_crawl4ai(url))
                if crawl4ai_result and crawl4ai_result.success:
                    return crawl4ai_result
            except Exception as e:
                logger.warning(f"Crawl4AI attempt failed for {url}: {str(e)}")

        # Fall back to traditional methods
        logger.info(f"Falling back to traditional scraping methods for {url}")

        # Rotate user agent for each request to reduce blocking
        self.session.headers.update({'User-Agent': ua.random})

        # Try extraction methods in order of preference
        extraction_methods = [
            self._extract_with_newspaper,
            self._extract_with_readability,
            self._extract_with_beautifulsoup
        ]

        last_error = None
        for method in extraction_methods:
            try:
                result = method(url)
                if result and result.success:
                    logger.info(f"Successfully scraped {url} using {result.metadata.get('method', 'unknown')}")
                    return result
            except Exception as e:
                last_error = str(e)
                logger.error(f"Method {method.__name__} failed for {url}: {str(e)}")
                continue

        # All methods failed - provide detailed error message
        error_message = "All extraction methods failed"
        if "403" in str(last_error) or "Forbidden" in str(last_error):
            error_message = "Access denied (403 Forbidden) - site may be blocking automated requests"
        elif "404" in str(last_error):
            error_message = "Page not found (404)"
        elif "timeout" in str(last_error).lower():
            error_message = "Request timeout - site may be slow or unreachable"
        elif last_error:
            error_message = f"All extraction methods failed: {last_error}"

        logger.error(f"All extraction methods failed for {url}: {error_message}")
        return ScrapedContent(
            url=url,
            title="Scraping Failed",
            content="",
            content_length=0,
            success=False,
            error_message=error_message
        )

    async def scrape_multiple_sources_async(self, sources: List[WebSearchResult]) -> List[ScrapedContent]:
        """
        Scrape content from multiple sources using Crawl4AI async capabilities.
        """
        if CRAWL4AI_AVAILABLE and len(sources) > 1:
            try:
                logger.info(f"Using Crawl4AI async scraping for {len(sources)} sources")
                scraper = get_crawl4ai_scraper()
                return await scraper.scrape_multiple_sources(sources)
            except Exception as e:
                logger.warning(f"Crawl4AI async scraping failed: {str(e)}, falling back to sync")

        # Fallback to sync scraping
        return self.scrape_multiple_sources(sources)

    def scrape_multiple_sources(self, sources: List[WebSearchResult]) -> List[ScrapedContent]:
        """
        Scrape content from multiple sources with rate limiting.
        For better performance with multiple sources, consider using scrape_multiple_sources_async.
        """
        results = []

        logger.info(f"Starting to scrape {len(sources)} sources")

        for i, source in enumerate(sources):
            logger.info(f"Scraping source {i+1}/{len(sources)}: {source.url}")

            # Rate limiting - respect delays between requests
            if i > 0:
                time.sleep(self.request_delay)

            result = self.scrape_content(source.url)
            results.append(result)

            # Log progress
            success_count = sum(1 for r in results if r.success)
            logger.info(f"Progress: {i+1}/{len(sources)}, Success rate: {success_count}/{i+1}")

        success_count = sum(1 for r in results if r.success)
        logger.info(f"Scraping completed. Success rate: {success_count}/{len(sources)}")

        return results


# Global scraper instance
_scraper = None

def get_scraper() -> WebScraper:
    """Get a global scraper instance."""
    global _scraper
    if _scraper is None:
        # Use settings if available
        delay = getattr(settings, 'SCRAPING_DELAY', 2.0)
        timeout = getattr(settings, 'SCRAPING_TIMEOUT', 30)
        _scraper = WebScraper(request_delay=delay, timeout=timeout)
    return _scraper


def scrape_content(url: str) -> ScrapedContent:
    """
    Scrape content from a single URL.

    Tries Crawl4AI first for best results, then falls back to newspaper3k,
    readability, and BeautifulSoup. Handles different content types and
    implements rate limiting.
    """
    scraper = get_scraper()
    return scraper.scrape_content(url)


def scrape_multiple_sources(sources: List[WebSearchResult]) -> List[ScrapedContent]:
    """
    Scrape content from multiple sources.

    Uses sync scraping with rate limiting. For better performance with
    multiple sources, consider using scrape_multiple_sources_async.
    """
    scraper = get_scraper()
    return scraper.scrape_multiple_sources(sources)


async def scrape_multiple_sources_async(sources: List[WebSearchResult]) -> List[ScrapedContent]:
    """
    Scrape content from multiple sources using async Crawl4AI capabilities.

    This is significantly faster than the sync version for multiple URLs.
    Falls back to sync scraping if Crawl4AI is not available.
    """
    scraper = get_scraper()
    return await scraper.scrape_multiple_sources_async(sources)


async def scrape_content_async(url: str) -> ScrapedContent:
    """
    Scrape content from a single URL using async Crawl4AI.

    More efficient than the sync version, especially for JavaScript-heavy sites
    and PDF files. Falls back to sync scraping if Crawl4AI is not available.
    """
    if CRAWL4AI_AVAILABLE:
        try:
            scraper = get_crawl4ai_scraper()
            return await scraper.scrape_content(url)
        except Exception as e:
            logger.warning(f"Async scraping failed for {url}: {str(e)}, falling back to sync")

    # Fallback to sync scraping
    return scrape_content(url)


async def scrape_pdf_async(url: str) -> ScrapedContent:
    """
    Scrape content from a PDF file using Crawl4AI.

    This function is specifically optimized for PDF files and provides
    better extraction than traditional web scraping methods.

    Args:
        url: URL of the PDF file to scrape

    Returns:
        ScrapedContent object with PDF text content
    """
    if not CRAWL4AI_AVAILABLE:
        logger.error("Crawl4AI is required for PDF scraping but not available")
        return ScrapedContent(
            url=url,
            title="PDF Scraping Failed",
            content="",
            content_length=0,
            success=False,
            error_message="Crawl4AI not available - PDF scraping requires Crawl4AI"
        )

    try:
        scraper = get_crawl4ai_scraper()
        return await scraper.scrape_pdf(url)
    except Exception as e:
        logger.error(f"PDF scraping failed for {url}: {str(e)}")
        return ScrapedContent(
            url=url,
            title="PDF Scraping Failed",
            content="",
            content_length=0,
            success=False,
            error_message=f"PDF scraping error: {str(e)}"
        )
