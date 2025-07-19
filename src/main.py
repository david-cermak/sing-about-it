"""
Enhanced Learning Sheet Generator - Main Orchestrator

This is the main entry point for the enhanced learning sheet generator.
It orchestrates the entire pipeline from search to final sheet generation.

Architecture:
- Phase 1: ✅ Modular structure with configuration management
- Phase 2: ⏳ Source evaluation agent (placeholder)
- Phase 3: ⏳ Web scraping implementation (placeholder)
- Phase 4: ⏳ Enhanced LLM generation (placeholder)
"""

from typing import Dict, List
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from config.models import WebSearchResult, LearningSheet
from search.web_search import (
    perform_comprehensive_search,
    format_search_results,
    check_duckduckgo_availability
)
from agents.sheet_generator import generate_learning_sheet_from_snippets
from utils.logging_utils import Timer, log_phase_start, log_phase_end, log_success, log_info


def display_system_status():
    """Display system status and configuration."""
    print("🚀 Enhanced Learning Sheet Generator")
    print("="*60)

    print("🔧 System Configuration:")
    print(f"  🤖 LLM Model: {settings.llm.model}")
    print(f"  🌐 Base URL: {settings.llm.base_url}")
    print(f"  🔍 Search queries per topic: {settings.search.queries_per_topic}")
    print(f"  📊 Results per query: {settings.search.results_per_query}")
    print(f"  🎯 Max sources to scrape: {settings.source_selection.max_sources_to_scrape}")

    print("\n🔧 System Check:")
    if check_duckduckgo_availability():
        print("  ✅ DuckDuckGo search is available - using real web search")
    else:
        print("  ⚠️  DuckDuckGo search not available - using fallback results")

    print("\n📋 Current Implementation Status:")
    print("  ✅ Phase 1: Modular architecture with configuration")
    print("  ⏳ Phase 2: Source evaluation (placeholder)")
    print("  ⏳ Phase 3: Web scraping (placeholder)")
    print("  ⏳ Phase 4: Enhanced generation (placeholder)")
    print("="*60)


def search_and_gather_sources(topic: str) -> Dict[str, List[WebSearchResult]]:
    """Phase 1: Search and gather sources."""
    timer = Timer()
    timer.start()

    log_phase_start("Web Search")
    log_info(f"Topic: '{topic}'")

    # Perform comprehensive search
    search_results = perform_comprehensive_search(topic)

    duration = timer.stop()
    log_phase_end("Web Search", duration)

    return search_results


def evaluate_and_select_sources(search_results: Dict[str, List[WebSearchResult]], topic: str):
    """Phase 2: Evaluate and select sources (placeholder)."""
    log_phase_start("Source Evaluation")
    log_info("Using placeholder source evaluation (to be implemented in Phase 2)")

    # For now, just count the sources
    all_sources = []
    for results in search_results.values():
        all_sources.extend(results)

    log_info(f"Found {len(all_sources)} total sources")
    log_info(f"Will use top {min(len(all_sources), settings.source_selection.max_sources_to_scrape)} sources for scraping")
    log_phase_end("Source Evaluation", 0.1)  # Placeholder timing


def scrape_content_from_sources():
    """Phase 3: Scrape content from selected sources (placeholder)."""
    log_phase_start("Content Scraping")
    log_info("Content scraping not yet implemented (Phase 3)")
    log_info("Currently using search snippets instead of full content")
    log_phase_end("Content Scraping", 0.1)  # Placeholder timing


def generate_learning_sheet(topic: str, search_results: Dict[str, List[WebSearchResult]]) -> LearningSheet:
    """Phase 4: Generate learning sheet."""
    timer = Timer()
    timer.start()

    log_phase_start("Learning Sheet Generation")

    # Format search results for the AI agent
    formatted_results = format_search_results(search_results)

    if settings.logging.verbose_output:
        print("📋 Formatted search results for LLM:")
        print(f"   Total characters: {len(formatted_results):,}")
        print(f"   Number of search queries: {len(search_results)}")
        print()

    # Create prompt and display it
    if settings.logging.verbose_output:
        print("🤖 Generating learning sheet with current implementation...")
        print("   (Using search snippets - full content scraping in Phase 3)")
        print()

    # Generate the learning sheet
    try:
        sheet = generate_learning_sheet_from_snippets(topic, formatted_results)
        duration = timer.stop()
        log_phase_end("Learning Sheet Generation", duration)
        return sheet

    except Exception as e:
        log_info(f"❌ Error generating learning sheet: {e}")
        log_info("This might be due to the AI model response format. Please try again.")
        raise


def display_results(sheet: LearningSheet, search_results: Dict[str, List[WebSearchResult]]):
    """Display the final results."""
    print("✅ Learning sheet generated successfully!")
    print("="*60)
    print(f"# {sheet.title}")
    print("="*60)
    print(sheet.content)

    # Display sources used
    print("\n" + "="*60)
    print("📚 SOURCES REFERENCED:")
    print("="*60)

    source_count = 0
    for query, results in search_results.items():
        if results:  # Only show if there are results
            print(f"\n🔍 From search: '{query}'")
            for i, result in enumerate(results, 1):
                source_count += 1
                print(f"  {source_count}. {result.title}")
                print(f"     🌐 {result.url}")

    print(f"\n📊 Total sources used: {source_count}")
    print("="*60)


def main():
    """Main application entry point."""
    # Display system status
    display_system_status()

    # Get topic from user
    topic = input("\n📝 Enter a topic for the learning sheet: ")

    try:
        # Phase 1: Search and gather sources
        search_results = search_and_gather_sources(topic)

        # Phase 2: Evaluate and select sources (placeholder)
        evaluate_and_select_sources(search_results, topic)

        # Phase 3: Scrape content from sources (placeholder)
        scrape_content_from_sources()

        # Phase 4: Generate learning sheet
        sheet = generate_learning_sheet(topic, search_results)

        # Display results
        display_results(sheet, search_results)

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if settings.logging.verbose_output:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
