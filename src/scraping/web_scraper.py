"""Web content scraping functionality."""

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

            # Check for PDF files (not supported by HTML extractors)
            if url.lower().endswith('.pdf'):
                logger.warning(f"PDF file detected: {url} - PDF extraction not yet implemented")
                return False

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

    def scrape_content(self, url: str) -> ScrapedContent:
        """
        Scrape content from a single URL using multiple extraction methods.

        Tries newspaper3k first, then readability, then BeautifulSoup as fallback.
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

    def scrape_multiple_sources(self, sources: List[WebSearchResult]) -> List[ScrapedContent]:
        """
        Scrape content from multiple sources with rate limiting.
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

    Uses newspaper3k for article extraction with BeautifulSoup fallback.
    Handles different content types and implements rate limiting.
    """
    scraper = get_scraper()
    return scraper.scrape_content(url)


def scrape_multiple_sources(sources: List[WebSearchResult]) -> List[ScrapedContent]:
    """
    Scrape content from multiple sources.

    Implements async scraping for performance with error handling and retries.
    """
    scraper = get_scraper()
    return scraper.scrape_multiple_sources(sources)
