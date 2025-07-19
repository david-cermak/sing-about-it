"""Learning sheet generation agent."""

from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import LearningSheet, EnhancedLearningSheet, ScrapedContent
from config.settings import settings


def create_sheet_generator() -> Agent:
    """
    Create and configure the learning sheet generation agent.

    For now, this uses the existing LearningSheet model.
    TODO: Migrate to EnhancedLearningSheet in Phase 4
    """
    model = OpenAIModel(
        model_name=settings.llm.model,
        provider=OpenAIProvider(base_url=settings.llm.base_url)
    )

    return Agent(
        model,
        output_type=LearningSheet,
        system_prompt=(
            "You are an educational content creator with access to current web information. "
            "You must return a JSON response with exactly two fields: 'title' (string) and 'content' (string). "
            "The title should be concise and descriptive. "
            "The content should be a comprehensive ~1000 word detailed report in Markdown format. "
            "Use the web search results to provide up-to-date information, examples, and insights. "
            "Include references to sources where appropriate. "
            "Structure the content with clear headings, bullet points, and examples. "
            "Focus on practical applications and current relevance of the topic. "
            "Always return valid JSON with only 'title' and 'content' fields."
        )
    )


def generate_learning_sheet_from_snippets(topic: str, formatted_search_results: str) -> LearningSheet:
    """
    Generate learning sheet using search snippets (current implementation).

    This is the current implementation migrated from the original file.
    TODO: Replace with generate_learning_sheet_from_content in Phase 4
    """
    agent = create_sheet_generator()

    prompt = f"""Please create a comprehensive learning sheet about: {topic}

{formatted_search_results}

Use the above web search results to create an informative, up-to-date learning sheet.
Include relevant examples, current trends, and practical applications from the search results.
Make sure to reference credible sources and provide actionable insights.

Return your response as a JSON object with 'title' and 'content' fields only."""

    result = agent.run_sync(prompt)
    return result.data


def generate_learning_sheet_from_content(topic: str, scraped_contents: List[ScrapedContent]) -> EnhancedLearningSheet:
    """
    Generate enhanced learning sheet using full scraped content.

    TODO: Implement in Phase 4
    - Use full content instead of snippets
    - Generate enhanced output with metadata
    - Include proper citations and confidence scores
    """
    # Placeholder implementation
    raise NotImplementedError("Enhanced learning sheet generation will be implemented in Phase 4")
