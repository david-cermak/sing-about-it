"""
Learning Sheet Generator with Web Search Integration

This script generates educational learning sheets enhanced with web search results.
The current implementation uses DuckDuckGo search for real-time web results.

DuckDuckGo Search Setup:
- Install: pip install --user duckduckgo-search
- No API key required
- Documentation: https://github.com/deedy5/duckduckgo_search

The script will automatically fall back to structured mock results if DuckDuckGo search is not available.

Alternative web search APIs that can be integrated:
1. Google Search API: pip install google-api-python-client
2. Bing Web Search API: pip install azure-cognitiveservices-search-websearch  
3. SerpAPI: pip install google-search-results
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import requests
from typing import List, Dict
import re

class LearningSheet(BaseModel):
    title: str = Field(description="Title of the learning sheet")
    content: str = Field(description="~1000 word report in Markdown")

class WebSearchResult(BaseModel):
    url: str
    title: str
    snippet: str

def search_web(query: str, num_results: int = 5) -> List[WebSearchResult]:
    """
    Search the web for information about a topic using DuckDuckGo search.
    """
    try:
        # Try to import and use DuckDuckGo search
        from duckduckgo_search import DDGS
        
        search_results = []
        
        # Perform DuckDuckGo search
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            
            for result in results:
                search_results.append(WebSearchResult(
                    url=result.get('href', ''),
                    title=result.get('title', ''),
                    snippet=result.get('body', '')
                ))
        
        return search_results
        
    except ImportError:
        print("DuckDuckGo search not available. Please install: pip install --user duckduckgo-search")
        # Fallback to structured mock results
        return _get_fallback_results(query, num_results)
        
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        # Fallback to structured mock results
        return _get_fallback_results(query, num_results)

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
            snippet=f"{description} for {query}. This resource provides detailed information with practical applications and current insights."
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
    
    search_results = {}
    for query in search_queries:
        results = search_web(query, num_results=3)
        search_results[query] = results
    
    return search_results

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

ollama_model = OpenAIModel(
    model_name="llama3.2",
    provider=OpenAIProvider(base_url="http://172.17.16.1:11434/v1")
)

# Create PydanticAI agent with enhanced system prompt
agent = Agent(
    ollama_model,
    output_type=LearningSheet,
    system_prompt=(
        "You are an educational content creator with access to current web information. "
        "Given a topic and recent web search results, generate a comprehensive learning sheet: "
        "Title and ~1000 word detailed report in Markdown. "
        "Use the web search results to provide up-to-date information, examples, and insights. "
        "Include references to sources where appropriate. "
        "Structure the content with clear headings, bullet points, and examples. "
        "Focus on practical applications and current relevance of the topic."
    )
)

def check_duckduckgo_availability():
    """Check if DuckDuckGo search is available."""
    try:
        from duckduckgo_search import DDGS
        return True
    except ImportError:
        return False

def main():
    # Check DuckDuckGo search availability
    if check_duckduckgo_availability():
        print("✅ DuckDuckGo search is available - using real web search")
    else:
        print("⚠️  DuckDuckGo search not available - using fallback results")
        print("   To enable real web search, install: pip install --user duckduckgo-search")
    
    topic = input("\nEnter a topic for the learning sheet: ")
    
    print(f"🔍 Searching the web for information about '{topic}'...")
    
    # Perform comprehensive web search
    search_results = perform_comprehensive_search(topic)
    
    # Format search results for the AI agent
    formatted_results = format_search_results(search_results)
    
    print("📝 Generating learning sheet with web research...")
    
    # Create prompt with topic and search results
    prompt = f"""Please create a comprehensive learning sheet about: {topic}

{formatted_results}

Use the above web search results to create an informative, up-to-date learning sheet. 
Include relevant examples, current trends, and practical applications from the search results.
Make sure to reference credible sources and provide actionable insights."""

    result = agent.run_sync(prompt)
    sheet: LearningSheet = result.data

    print(f"\n# {sheet.title}\n")
    print(sheet.content)
    
    # Display sources used
    print("\n---\n## Sources Referenced:")
    for query, results in search_results.items():
        for result in results:
            print(f"- {result.title}: {result.url}")

if __name__ == "__main__":
    main()
