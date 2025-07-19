"""
Enhanced Learning Sheet Generator - Main Orchestrator

This is the main entry point for the enhanced learning sheet generator.
It orchestrates the entire pipeline from search to final sheet generation.

Architecture:
- Phase 1: ✅ Modular structure with configuration management
- Phase 2: ✅ Intelligent source evaluation with LLM agent
- Phase 3: ⏳ Web scraping implementation (placeholder)
- Phase 4: ⏳ Enhanced LLM generation (placeholder)
"""

from typing import Dict, List, Optional
import sys
import json
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from config.models import WebSearchResult, LearningSheet, SourceEvaluation
from search.web_search import (
    perform_comprehensive_search,
    format_search_results,
    check_duckduckgo_availability
)
from agents.sheet_generator import generate_learning_sheet_from_snippets
from agents.source_evaluator import evaluate_sources, select_top_sources
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
    print("  ✅ Phase 2: Intelligent source evaluation with LLM agent")
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


def evaluate_and_select_sources(search_results: Dict[str, List[WebSearchResult]], topic: str) -> List[WebSearchResult]:
    """Phase 2: Evaluate and select sources using intelligent LLM-based evaluation."""
    timer = Timer()
    timer.start()

    log_phase_start("Source Evaluation")

    # Flatten all search results into a single list
    all_sources = []
    for results in search_results.values():
        all_sources.extend(results)

    log_info(f"Found {len(all_sources)} total sources from search")

    if not all_sources:
        log_info("No sources found to evaluate")
        duration = timer.stop()
        log_phase_end("Source Evaluation", duration)
        return []

    # Evaluate each source using the LLM agent
    evaluations = evaluate_sources(all_sources, topic)

    # Select the best sources based on evaluation scores
    selected_evaluations = select_top_sources(evaluations)

    # Convert back to WebSearchResult objects for compatibility
    selected_sources = []
    for evaluation in selected_evaluations:
        # Find the original WebSearchResult
        for source in all_sources:
            if source.url == evaluation.url:
                selected_sources.append(source)
                break

    duration = timer.stop()
    log_info(f"Selected {len(selected_sources)} high-quality sources for scraping")
    log_phase_end("Source Evaluation", duration)

    return selected_sources


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
    sheet = generate_learning_sheet_from_snippets(topic, formatted_results)
    duration = timer.stop()
    log_phase_end("Learning Sheet Generation", duration)
    return sheet


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


def save_phase_results(phase_name: str, topic: str, data: dict):
    """Save phase results to JSON file."""
    filename = f"results_{phase_name}_{topic.replace(' ', '_')}.json"
    filepath = Path(filename)

    # Convert WebSearchResult objects to dicts for JSON serialization
    serializable_data = {}
    for key, value in data.items():
        if key in ['search_results', 'original_search_results']:
            # Handle search results (dict of query -> list of WebSearchResult)
            serializable_data[key] = {}
            for query, results in value.items():
                serializable_data[key][query] = [
                    {
                        'url': r.url,
                        'title': r.title,
                        'snippet': r.snippet,
                        'search_query': r.search_query
                    } for r in results
                ]
        else:
            serializable_data[key] = value
    data = serializable_data

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved {phase_name} results to: {filepath}")


def load_phase_results(phase_name: str, topic: str) -> Optional[dict]:
    """Load phase results from JSON file."""
    filename = f"results_{phase_name}_{topic.replace(' ', '_')}.json"
    filepath = Path(filename)

    if not filepath.exists():
        print(f"⚠️  No saved results found: {filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert back to WebSearchResult objects if needed
    from config.models import WebSearchResult

    # Handle both search_results and original_search_results
    for key in ['search_results', 'original_search_results']:
        if key in data:
            converted_results = {}
            for query, results in data[key].items():
                converted_results[query] = [
                    WebSearchResult(
                        url=r['url'],
                        title=r['title'],
                        snippet=r['snippet'],
                        search_query=r.get('search_query', query)
                    ) for r in results
                ]
            data[key] = converted_results

    print(f"📂 Loaded {phase_name} results from: {filepath}")
    return data


def run_phase_search(topic: str) -> Dict[str, List[WebSearchResult]]:
    """Run Phase 1: Search and gather sources."""
    print(f"\n🚀 Running Phase 1: Search for '{topic}'")
    search_results = search_and_gather_sources(topic)

    # Save results
    save_phase_results("search", topic, {
        "topic": topic,
        "search_results": search_results
    })

    return search_results


def run_phase_evaluation(topic: str, search_results: Optional[Dict[str, List[WebSearchResult]]] = None, input_file: Optional[str] = None) -> List[WebSearchResult]:
    """Run Phase 2: Source evaluation."""
    if search_results is None:
        if input_file:
            # Load from specified file
            print(f"📂 Loading search results from: {input_file}")
            if not Path(input_file).exists():
                print(f"❌ File not found: {input_file}")
                return []

            with open(input_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)

            # Convert back to WebSearchResult objects
            if 'search_results' in saved_data:
                from config.models import WebSearchResult
                search_results = {}
                for query, results in saved_data['search_results'].items():
                    search_results[query] = [
                        WebSearchResult(
                            url=r['url'],
                            title=r['title'],
                            snippet=r['snippet'],
                            search_query=r.get('search_query', query)
                        ) for r in results
                    ]
                # Update topic from file if not provided
                if 'topic' in saved_data and not topic:
                    topic = saved_data['topic']
            else:
                print("❌ Invalid search results file format")
                return []
        else:
            # Try to load from auto-generated filename
            saved_data = load_phase_results("search", topic)
            if saved_data is None:
                print("❌ No search results available. Run phase 1 first or specify --file.")
                return []
            search_results = saved_data['search_results']

    print(f"\n🚀 Running Phase 2: Source Evaluation for '{topic}'")
    selected_sources = evaluate_and_select_sources(search_results, topic)

    # Save results
    save_phase_results("evaluation", topic, {
        "topic": topic,
        "selected_sources": [
            {
                'url': s.url,
                'title': s.title,
                'snippet': s.snippet,
                'search_query': s.search_query
            } for s in selected_sources
        ],
        "original_search_results": search_results
    })

    return selected_sources


def run_phase_generation(topic: str, search_results: Optional[Dict[str, List[WebSearchResult]]] = None, input_file: Optional[str] = None) -> LearningSheet:
    """Run Phase 4: Learning sheet generation."""
    if search_results is None:
        if input_file:
            # Load from specified file
            print(f"📂 Loading results from: {input_file}")
            if not Path(input_file).exists():
                print(f"❌ File not found: {input_file}")
                return None

            with open(input_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)

            # Determine file type and extract search results
            if 'original_search_results' in file_data:
                # Evaluation results file
                from config.models import WebSearchResult
                raw_search_results = file_data['original_search_results']
                search_results = {}
                for query, results in raw_search_results.items():
                    search_results[query] = [
                        WebSearchResult(
                            url=r['url'],
                            title=r['title'],
                            snippet=r['snippet'],
                            search_query=r.get('search_query', query)
                        ) for r in results
                    ]
            elif 'search_results' in file_data:
                # Search results file
                from config.models import WebSearchResult
                search_results = {}
                for query, results in file_data['search_results'].items():
                    search_results[query] = [
                        WebSearchResult(
                            url=r['url'],
                            title=r['title'],
                            snippet=r['snippet'],
                            search_query=r.get('search_query', query)
                        ) for r in results
                    ]
            else:
                print("❌ Invalid file format for generation phase")
                return None

            # Update topic from file if not provided
            if 'topic' in file_data and not topic:
                topic = file_data['topic']
        else:
            # Try to load from evaluation results first
            eval_data = load_phase_results("evaluation", topic)
            if eval_data and 'original_search_results' in eval_data:
                search_results = eval_data['original_search_results']
            else:
                # Fallback to search results
                search_data = load_phase_results("search", topic)
                if search_data is None:
                    print("❌ No search results available. Run phase 1 first or specify --file.")
                    return None
                search_results = search_data['search_results']

    print(f"\n🚀 Running Phase 4: Learning Sheet Generation for '{topic}'")

    # Debug the search_results structure
    if settings.logging.verbose_output:
        print(f"\n🔍 DEBUG: search_results type: {type(search_results)}")
        if search_results:
            first_query = next(iter(search_results.keys()))
            first_results = search_results[first_query]
            print(f"🔍 DEBUG: First query: {first_query}")
            print(f"🔍 DEBUG: First results type: {type(first_results)}")
            if first_results:
                print(f"🔍 DEBUG: First result type: {type(first_results[0])}")
                print(f"🔍 DEBUG: First result: {first_results[0]}")

    # Add debugging for this phase
    try:
        sheet = generate_learning_sheet(topic, search_results)

        # Save results
        save_phase_results("generation", topic, {
            "topic": topic,
            "title": sheet.title,
            "content": sheet.content
        })

        return sheet

    except Exception as e:
        print(f"\n❌ Learning Sheet Generation Failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")

        # Add direct debugging here
        print("\n🔍 DEBUGGING - Testing direct Ollama connection...")
        try:
            import requests

            base_url = settings.llm.base_url.replace('/v1', '')
            response = requests.get(f"{base_url}/api/tags", timeout=10)

            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"✅ Ollama connected. Models: {[m.get('name', '') for m in models]}")
            else:
                print(f"❌ Ollama connection failed: {response.status_code}")

        except Exception as conn_e:
            print(f"❌ Connection test failed: {conn_e}")

        raise


def main():
    """Main application entry point with phase control."""
    parser = argparse.ArgumentParser(description="Learning Sheet Generator")
    parser.add_argument('--phase', choices=['search', 'eval', 'generate', 'all'],
                       default='all', help='Which phase to run')
    parser.add_argument('--topic', type=str, help='Topic for learning sheet')
    parser.add_argument('--file', type=str, help='Input file to load (e.g., results_search_topic.json)')
    parser.add_argument('--list-saves', action='store_true', help='List saved result files')

    args = parser.parse_args()

    # List saved files
    if args.list_saves:
        result_files = list(Path('.').glob('results_*.json'))
        if result_files:
            print("💾 Saved result files:")
            for file in sorted(result_files):
                print(f"  📄 {file}")
        else:
            print("📂 No saved result files found")
        return

    # Display system status
    display_system_status()

    # Get topic
    topic = None
    if args.topic:
        topic = args.topic
        print(f"\n📝 Using topic from command line: '{topic}'")
    elif args.file:
        # Try to extract topic from file
        print(f"📂 Extracting topic from file: {args.file}")
        if Path(args.file).exists():
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                if 'topic' in file_data:
                    topic = file_data['topic']
                    print(f"📝 Found topic in file: '{topic}'")
                else:
                    print("⚠️  No topic found in file, will ask for input")
            except Exception as e:
                print(f"❌ Error reading file: {e}")
        else:
            print(f"❌ File not found: {args.file}")

    if not topic:
        topic = input("\n📝 Enter a topic for the learning sheet: ")

    try:
        search_results = None
        selected_sources = None
        sheet = None

        if args.phase in ['search', 'all']:
            search_results = run_phase_search(topic)

        if args.phase in ['eval', 'all']:
            selected_sources = run_phase_evaluation(topic, search_results, args.file)

        if args.phase == 'eval':
            if selected_sources:
                print(f"\n✅ Source evaluation complete. {len(selected_sources)} sources selected.")
            return

        if args.phase in ['generate', 'all']:
            # For generation, we need the original search results, not just selected sources
            if search_results is None:
                # Load from files
                eval_data = load_phase_results("evaluation", topic)
                if eval_data and 'original_search_results' in eval_data:
                    # The load_phase_results function should have already converted this
                    search_results = eval_data['original_search_results']
                    # Debug: Check if conversion worked
                    if settings.logging.verbose_output:
                        print(f"🔍 DEBUG: Loaded from eval_data, type: {type(search_results)}")
                        if search_results:
                            first_key = next(iter(search_results.keys()))
                            print(f"🔍 DEBUG: First result type: {type(search_results[first_key][0]) if search_results[first_key] else 'empty'}")
                else:
                    search_data = load_phase_results("search", topic)
                    if search_data:
                        search_results = search_data['search_results']

            sheet = run_phase_generation(topic, search_results, args.file)

            if sheet:
                # Display results
                display_results(sheet, search_results)

        if args.phase == 'search':
            print(f"\n✅ Search complete. Found {sum(len(results) for results in search_results.values())} total results.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if settings.logging.verbose_output:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
