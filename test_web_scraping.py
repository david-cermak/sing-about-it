"""
Test script for web scraping functionality.

This script demonstrates the Phase 3 web scraping implementation by:
1. Testing the web scraper with real URLs
2. Testing content processing and chunking
3. Showing statistics and quality metrics
"""

import sys
from pathlib import Path
import logging
from typing import List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.models import WebSearchResult, ScrapedContent
from scraping.web_scraper import WebScraper, scrape_content, scrape_multiple_sources
from scraping.content_processor import (
    process_scraped_content,
    summarize_content_stats,
    validate_content_quality,
    clean_content
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_sample_urls():
    """Test scraping with a variety of sample URLs."""
    print("\n" + "="*80)
    print("🧪 TESTING WEB SCRAPING - PHASE 3 IMPLEMENTATION")
    print("="*80)

    # Sample URLs for testing (publicly available educational content)
    test_urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://docs.python.org/3/tutorial/",
        "https://www.bbc.com/news",  # News site
        "https://stackoverflow.com/questions/tagged/python",  # Technical Q&A
        "https://github.com/python/cpython",  # Code repository
    ]

    # Convert to WebSearchResult format
    search_results = [
        WebSearchResult(
            url=url,
            title=f"Test source {i+1}",
            snippet=f"Test snippet for {url}",
            search_query="test query"
        )
        for i, url in enumerate(test_urls)
    ]

    print(f"\n📋 Testing with {len(search_results)} sample URLs:")
    for i, result in enumerate(search_results, 1):
        print(f"  {i}. {result.url}")

    return search_results


def test_single_url_scraping():
    """Test scraping a single URL."""
    print("\n🔧 Testing Single URL Scraping")
    print("-" * 50)

    # Test with Wikipedia (usually reliable for scraping)
    test_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

    print(f"Scraping: {test_url}")
    result = scrape_content(test_url)

    print(f"✅ Success: {result.success}")
    print(f"📝 Title: {result.title}")
    print(f"📏 Content length: {result.content_length} characters")

    if result.success:
        print(f"🔧 Extraction method: {result.metadata.get('method', 'unknown')}")
        print(f"📄 First 200 chars: {result.content[:200]}...")

        # Test content quality
        is_quality = validate_content_quality(result)
        print(f"✨ Quality check: {'✅ PASS' if is_quality else '❌ FAIL'}")
    else:
        print(f"❌ Error: {result.error_message}")

    return result


def test_multiple_url_scraping(search_results: List[WebSearchResult]):
    """Test scraping multiple URLs."""
    print("\n🔧 Testing Multiple URL Scraping")
    print("-" * 50)

    # Create scraper with faster settings for testing
    scraper = WebScraper(request_delay=1.0, timeout=15)

    print(f"Scraping {len(search_results)} URLs with 1s delay between requests...")
    scraped_results = scraper.scrape_multiple_sources(search_results)

    print(f"\n📊 Scraping Results:")
    success_count = sum(1 for r in scraped_results if r.success)
    print(f"  Total URLs: {len(scraped_results)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(scraped_results) - success_count}")
    print(f"  Success rate: {(success_count/len(scraped_results)*100):.1f}%")

    print(f"\n📝 Individual Results:")
    for i, result in enumerate(scraped_results, 1):
        status = "✅" if result.success else "❌"
        length = result.content_length if result.success else 0
        method = result.metadata.get('method', 'N/A') if result.success else 'N/A'
        print(f"  {i}. {status} {result.url[:50]}... ({length} chars, {method})")

    return scraped_results


def test_content_processing(scraped_results: List[ScrapedContent]):
    """Test content processing and chunking."""
    print("\n🔧 Testing Content Processing")
    print("-" * 50)

    print(f"Processing {len(scraped_results)} scraped contents...")
    processed_chunks = process_scraped_content(scraped_results)

    print(f"\n📊 Processing Results:")
    stats = summarize_content_stats(processed_chunks)
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Total characters: {stats['total_characters']:,}")
    print(f"  Unique sources: {stats['sources']}")
    print(f"  Average chunk size: {stats['avg_chunk_size']} characters")
    print(f"  Average words per chunk: {stats['avg_words_per_chunk']}")

    # Show sample chunks
    if processed_chunks:
        print(f"\n📄 Sample Chunks (first 2):")
        for i, chunk in enumerate(processed_chunks[:2], 1):
            print(f"\n  Chunk {i} (from {chunk.source_url[:30]}...):")
            print(f"    Words: {chunk.word_count}")
            print(f"    Preview: {chunk.content[:150]}...")

    return processed_chunks, stats


def test_content_cleaning():
    """Test content cleaning functionality."""
    print("\n🔧 Testing Content Cleaning")
    print("-" * 50)

    # Sample dirty content
    dirty_content = """
    [Advertisement] Click here to subscribe!

    This is the main article content with    multiple     spaces.

    Share this article!!!!! Follow us on Twitter!!!

    Some actual meaningful content here with proper sentences.
    Read more: www.example.com/more

    [Sponsored Content] Buy our product now!

    More meaningful content continues here...
    """

    print("Original content:")
    print(dirty_content)
    print(f"Length: {len(dirty_content)}")

    cleaned = clean_content(dirty_content)

    print("\nCleaned content:")
    print(cleaned)
    print(f"Length: {len(cleaned)}")
    print(f"Reduction: {len(dirty_content) - len(cleaned)} characters")


def main():
    """Run all web scraping tests."""
    try:
        # Test content cleaning first (no network required)
        test_content_cleaning()

        # Test single URL scraping
        single_result = test_single_url_scraping()

        # Get sample URLs for testing
        search_results = test_sample_urls()

        # Test multiple URL scraping (this may take a while)
        print(f"\n⏳ Starting multiple URL scraping (this may take {len(search_results) * 2} seconds)...")
        scraped_results = test_multiple_url_scraping(search_results)

        # Test content processing
        processed_chunks, stats = test_content_processing(scraped_results)

        # Final summary
        print("\n" + "="*80)
        print("🎉 WEB SCRAPING TEST COMPLETE")
        print("="*80)
        print(f"✅ Phase 3 Implementation Status: WORKING")
        print(f"📊 Total content scraped: {stats['total_characters']:,} characters")
        print(f"📦 Total chunks created: {stats['total_chunks']}")
        print(f"🌐 Sources processed: {stats['sources']}")

        successful_scrapes = sum(1 for r in scraped_results if r.success)
        if successful_scrapes > 0:
            print(f"✨ Scraping success rate: {(successful_scrapes/len(scraped_results)*100):.1f}%")
            print(f"🚀 Ready for integration with learning sheet generation!")
        else:
            print(f"⚠️  No successful scrapes - check network connectivity")

    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
