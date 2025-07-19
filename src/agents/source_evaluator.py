"""Source evaluation agent for selecting high-quality sources."""

from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import WebSearchResult, SourceEvaluation
from config.settings import settings


# Create the source evaluation agent
def create_source_evaluator() -> Agent:
    """
    Create and configure the source evaluation agent.

    TODO: Implement in Phase 2
    - Create agent with proper system prompt
    - Define evaluation criteria
    - Configure output format
    """
    model = OpenAIModel(
        model_name=settings.llm.model,
        provider=OpenAIProvider(base_url=settings.llm.base_url)
    )

    return Agent(
        model,
        output_type=SourceEvaluation,
        system_prompt="Source evaluation agent - to be implemented in Phase 2"
    )


def evaluate_sources(search_results: List[WebSearchResult], topic: str) -> List[SourceEvaluation]:
    """
    Evaluate search results and select the best sources for scraping.

    TODO: Implement in Phase 2
    - Analyze relevance to topic
    - Check domain authority
    - Assess content type and quality
    - Score and rank sources
    """
    # Placeholder implementation
    evaluations = []
    for result in search_results:
        evaluation = SourceEvaluation(
            url=result.url,
            title=result.title,
            relevance_score=0.8,  # Placeholder
            authority_score=0.7,  # Placeholder
            content_type="article",
            should_scrape=True,
            reasoning="Placeholder evaluation - to be implemented in Phase 2",
            estimated_quality="medium"
        )
        evaluations.append(evaluation)

    return evaluations


def select_top_sources(evaluations: List[SourceEvaluation]) -> List[SourceEvaluation]:
    """
    Select the top sources based on evaluation scores.

    TODO: Implement in Phase 2
    - Filter by minimum relevance score
    - Sort by combined relevance and authority scores
    - Limit to maximum number of sources to scrape
    """
    # Placeholder implementation
    selected = [e for e in evaluations if e.should_scrape]
    return selected[:settings.source_selection.max_sources_to_scrape]
