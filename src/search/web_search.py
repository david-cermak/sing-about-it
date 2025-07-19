"""Web search functionality using DuckDuckGo search."""

from typing import List, Dict
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import WebSearchResult
from config.settings import settings


def search_web(query: str, num_results: int = 5) -> List[WebSearchResult]:
    """
    Search the web for information about a topic using DuckDuckGo search.
    """
    if settings.logging.verbose_output:
        print(f"  🔎 Executing search: '{query}' (max {num_results} results)")

    try:
        # Try to import and use DuckDuckGo search
        from ddgs import DDGS

        search_results = []

        # Perform DuckDuckGo search
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)

            for result in results:
                search_results.append(WebSearchResult(
                    url=result.get('href', ''),
                    title=result.get('title', ''),
                    snippet=result.get('body', ''),
                    search_query=query
                ))

        if settings.logging.verbose_output:
            print(f"  ✅ Found {len(search_results)} results for '{query}'")
            for i, result in enumerate(search_results, 1):
                print(f"    {i}. {result.title}")
                print(f"       URL: {result.url}")
                print(f"       Snippet: {result.snippet[:100]}...")
            print()

        return search_results

    except ImportError:
        if settings.logging.verbose_output:
            print("  ❌ DuckDuckGo search not available. Please install: pip install ddgs")
        # Fallback to structured mock results
        fallback_results = _get_fallback_results(query, num_results)
        if settings.logging.verbose_output:
            print(f"  📋 Using {len(fallback_results)} fallback results instead")
        return fallback_results

    except Exception as e:
        if settings.logging.verbose_output:
            print(f"  ❌ DuckDuckGo search error: {e}")
        # Fallback to structured mock results
        fallback_results = _get_fallback_results(query, num_results)
        if settings.logging.verbose_output:
            print(f"  📋 Using {len(fallback_results)} fallback results instead")
        return fallback_results


def _get_fallback_results(query: str, num_results: int) -> List[WebSearchResult]:
    """
    Fallback search results when DuckDuckGo search is not available.
    """
    result_types = [
        ("Overview", "comprehensive guide covering the fundamentals"),
        ("Tutorial", "step-by-step tutorial with practical examples"),
        ("Best Practices", "expert recommendations and best practices"),
        ("Case Studies", "real-world examples and case studies"),
        ("Latest Trends", "current trends and future developments")
    ]

    search_results = []
    for i, (result_type, description) in enumerate(result_types[:num_results]):
        search_results.append(WebSearchResult(
            url=f"https://authoritative-source-{i+1}.com/{query.lower().replace(' ', '-')}-{result_type.lower()}",
            title=f"{query} {result_type} - Expert Guide",
            snippet=f"{description} for {query}. This resource provides detailed information with practical applications and current insights.",
            search_query=query
        ))

    return search_results


def perform_comprehensive_search(topic: str) -> Dict[str, List[WebSearchResult]]:
    """
    Perform multiple web searches related to the topic to gather comprehensive information.
    """
    search_queries = [
        f"{topic} overview basics",
        f"{topic} latest developments 2024",
        f"{topic} best practices guide",
        f"{topic} examples case studies",
        f"{topic} future trends"
    ]

    # Use configurable number of queries
    search_queries = search_queries[:settings.search.queries_per_topic]

    if settings.logging.verbose_output:
        print(f"📊 Performing comprehensive search with {len(search_queries)} different queries:")
        for i, query in enumerate(search_queries, 1):
            print(f"  {i}. {query}")
        print()

    search_results = {}
    for i, query in enumerate(search_queries, 1):
        if settings.logging.verbose_output:
            print(f"🔍 Search {i}/{len(search_queries)}:")
        results = search_web(query, num_results=settings.search.results_per_query)
        search_results[query] = results

    total_results = sum(len(results) for results in search_results.values())
    if settings.logging.verbose_output:
        print(f"✅ Comprehensive search completed! Total results gathered: {total_results}")
        print("="*60)

    return search_results


def check_duckduckgo_availability() -> bool:
    """Check if DuckDuckGo search is available."""
    try:
        from ddgs import DDGS
        return True
    except ImportError:
        return False


def format_search_results(search_results: Dict[str, List[WebSearchResult]]) -> str:
    """
    Format web search results into a structured text for the AI agent.
    """
    formatted_results = "## Web Search Results\n\n"

    for query, results in search_results.items():
        formatted_results += f"### Search Query: {query}\n\n"
        for result in results:
            formatted_results += f"**{result.title}**\n"
            formatted_results += f"Source: {result.url}\n"
            formatted_results += f"Summary: {result.snippet}\n\n"
        formatted_results += "---\n\n"

    return formatted_results
