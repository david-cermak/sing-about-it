# Crawl4AI Integration Guide

## 🚀 Overview

The scraping system has been enhanced with [Crawl4AI](https://github.com/unclecode/crawl4ai), a modern, AI-ready web scraping library that provides significant improvements over traditional scraping methods.

## ✨ Benefits of Crawl4AI

### **Superior Content Quality**
- **🤖 AI-Ready Markdown**: Generates clean, LLM-optimized markdown output
- **🎯 Smart Content Filtering**: Automatically removes ads, navigation, and boilerplate
- **📝 Better Text Extraction**: More accurate content extraction from complex layouts

### **Modern Web Support**
- **⚡ JavaScript Rendering**: Full Playwright-based browser automation
- **🔄 Dynamic Content**: Handles SPAs, infinite scroll, lazy loading
- **🛡️ Anti-Detection**: Better stealth capabilities than basic HTTP requests

### **Performance & Reliability**
- **🚀 Async Processing**: Built on asyncio for high-throughput scraping
- **⚙️ Concurrent Scraping**: Multiple URLs processed simultaneously
- **🎛️ Smart Caching**: Advanced caching mechanisms
- **🔁 Graceful Fallback**: Falls back to traditional methods if needed

## 🔧 Installation

### Prerequisites
```bash
# Install Crawl4AI
pip install crawl4ai

# Setup Playwright browsers
crawl4ai-setup
# OR manually:
# python -m playwright install --with-deps chromium
```

### Verification
```python
# Test if installation is working
python test_crawl4ai_integration.py
```

## 📖 Usage

### Basic Scraping

```python
from scraping.web_scraper import scrape_content, scrape_content_async

# Sync scraping (tries Crawl4AI first, falls back to traditional methods)
result = scrape_content("https://example.com")

# Async scraping (faster, more efficient)
result = await scrape_content_async("https://example.com")
```

### Multiple URLs

```python
from scraping.web_scraper import scrape_multiple_sources_async
from config.models import WebSearchResult

sources = [
    WebSearchResult(url="https://example1.com", title="Example 1", snippet="..."),
    WebSearchResult(url="https://example2.com", title="Example 2", snippet="..."),
]

# Concurrent scraping (much faster than sequential)
results = await scrape_multiple_sources_async(sources)
```

### Advanced Features

```python
from scraping.crawl4ai_scraper import get_crawl4ai_scraper

scraper = get_crawl4ai_scraper()

# Structured data extraction
css_schema = {
    "name": "ArticleExtractor",
    "baseSelector": "article",
    "fields": [
        {"name": "title", "selector": "h1", "type": "text"},
        {"name": "content", "selector": ".content", "type": "text"},
        {"name": "author", "selector": ".author", "type": "text"}
    ]
}

result = await scraper.scrape_with_extraction("https://blog.example.com", css_schema)

# Dynamic content handling
result = await scraper.scrape_dynamic_content(
    url="https://spa-example.com",
    wait_for_selector=".content-loaded",
    scroll_to_bottom=True,
    click_selectors=[".load-more-button"]
)
```

## 🏗️ Architecture

### Layered Approach
1. **Primary**: Crawl4AI (if available and working)
2. **Fallback**: Traditional methods (newspaper3k → readability → BeautifulSoup)

### Integration Points
- `WebScraper.scrape_content()` - Enhanced with Crawl4AI as primary method
- `scrape_content_async()` - New async interface using Crawl4AI
- `scrape_multiple_sources_async()` - Concurrent processing with Crawl4AI

### Backwards Compatibility
- All existing code continues to work unchanged
- Traditional methods still available as fallbacks
- Progressive enhancement approach

## ⚙️ Configuration

### Browser Settings
```python
from scraping.crawl4ai_scraper import Crawl4AIScraper

scraper = Crawl4AIScraper(
    headless=True,              # Run without GUI
    use_stealth=True,           # Anti-detection features
    request_delay=1.0,          # Delay between requests
    max_concurrent=5,           # Max concurrent requests
    timeout=30                  # Request timeout
)
```

### Content Filtering
```python
from crawl4ai.content_filter_strategy import PruningContentFilter

# Configure content filtering
content_filter = PruningContentFilter(
    threshold=0.48,             # Keep more content
    threshold_type="fixed",
    min_word_threshold=20       # Minimum words per block
)
```

## 🧪 Testing

### Integration Test
```bash
# Run comprehensive test suite
python test_crawl4ai_integration.py
```

### Manual Testing
```python
import asyncio
from scraping.web_scraper import scrape_content_async

async def test():
    result = await scrape_content_async("https://httpbin.org/html")
    print(f"Success: {result.success}")
    print(f"Content length: {result.content_length}")
    print(f"Method: {result.metadata.get('method')}")

asyncio.run(test())
```

## 🔍 Troubleshooting

### Common Issues

**1. Import Error**
```
ImportError: crawl4ai not available
```
**Solution**: Install crawl4ai: `pip install crawl4ai`

**2. Browser Setup Issues**
```
playwright._impl._api_types.Error: Executable doesn't exist
```
**Solution**: Run browser setup: `crawl4ai-setup` or `python -m playwright install --with-deps chromium`

**3. Async Context Issues**
```
RuntimeError: There is no current event loop
```
**Solution**: Use proper async context or the sync fallback methods

### Performance Tips

1. **Use Async Methods**: Much faster for multiple URLs
2. **Configure Concurrency**: Adjust `max_concurrent` based on target sites
3. **Enable Caching**: Reduces redundant requests
4. **Tune Delays**: Balance politeness vs speed

### Debugging

```python
# Enable verbose logging
import logging
logging.getLogger('crawl4ai').setLevel(logging.DEBUG)

# Test with headful browser
scraper = Crawl4AIScraper(headless=False)
```

## 📊 Performance Comparison

| Method | Speed | JavaScript | Content Quality | Resource Usage |
|--------|-------|------------|----------------|----------------|
| **Crawl4AI** | ⭐⭐⭐⭐⭐ | ✅ Full Support | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| newspaper3k | ⭐⭐⭐ | ❌ No | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| readability | ⭐⭐⭐ | ❌ No | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| BeautifulSoup | ⭐⭐ | ❌ No | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎯 Best Practices

1. **Prefer Async**: Use `scrape_content_async()` for better performance
2. **Batch Processing**: Use `scrape_multiple_sources_async()` for multiple URLs
3. **Configure Delays**: Be respectful to target websites
4. **Handle Failures**: Always check `result.success` and handle errors
5. **Monitor Resources**: Browser automation uses more memory
6. **Use Structured Extraction**: Leverage CSS selectors for consistent data

## 🔗 Related Documentation

- [Original Scraping README](README_SCRAPING.md)
- [Crawl4AI Official Docs](https://docs.crawl4ai.com/)
- [Content Processing Guide](content_processor.py)

## 📈 Future Enhancements

- [ ] LLM-based extraction strategies
- [ ] Deep crawling capabilities
- [ ] Session management for login-required sites
- [ ] PDF and document processing
- [ ] Custom extraction schemas
- [ ] Performance monitoring and metrics
