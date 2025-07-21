"""
Learning sheet generation agent using configurable backends.

This is the new implementation that supports both pydantic.ai and baremetal backends
while maintaining the same interface as the original sheet generator.
"""

from typing import List
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import LearningSheet, EnhancedLearningSheet, ScrapedContent
from config.settings import settings
from agents.factory import create_sheet_generator_backend
from agents.base import AgentBackend, APIError, ValidationError


class SheetGeneratorAgent:
    """
    Learning sheet generation agent with configurable backend support.

    This agent can use either pydantic.ai or baremetal backends based on
    configuration, providing flexibility while maintaining the same interface.
    """

    def __init__(self, backend: AgentBackend = None):
        """
        Initialize the sheet generator agent.

        Args:
            backend: Optional backend instance. If not provided, uses factory default.
        """
        self.backend = backend or create_sheet_generator_backend()

    def generate_from_snippets(self, topic: str, formatted_search_results: str) -> LearningSheet:
        """
        Generate learning sheet using search snippets.

        Args:
            topic: The topic to create a learning sheet for
            formatted_search_results: Formatted search results string

        Returns:
            LearningSheet with title and content
        """
        prompt = f"""Please create a comprehensive learning sheet about: {topic}

{formatted_search_results}

Use the above web search results to create an informative, up-to-date learning sheet.
Include relevant examples, current trends, and practical applications from the search results.
Make sure to reference credible sources and provide actionable insights.

Return your response as a JSON object with 'title' and 'content' fields only."""

        try:
            return self.backend.generate_response(prompt, LearningSheet)
        except Exception as e:
            if settings.logging.verbose_output:
                print(f"⚠️  Backend generation failed: {e}")

            # Fallback to simple content generation
            return self._fallback_generation(topic, formatted_search_results)

    def generate_from_content(self, topic: str, scraped_contents: List[ScrapedContent]) -> EnhancedLearningSheet:
        """
        Generate enhanced learning sheet using full scraped content.

        TODO: Implement in Phase 4
        - Use full content instead of snippets
        - Generate enhanced output with metadata
        - Include proper citations and confidence scores
        """
        # Placeholder implementation
        raise NotImplementedError("Enhanced learning sheet generation will be implemented in Phase 4")

    def _fallback_generation(self, topic: str, search_results: str) -> LearningSheet:
        """
        Simple fallback content generation if backend fails.
        """
        content = f"""# {topic}

## Overview
This learning sheet was generated based on available search results about {topic}.

## Key Information
{search_results[:1000]}...

## Summary
Due to technical limitations, this is a simplified learning sheet.
Please try again or check the system configuration.

## Sources
Based on web search results from multiple sources."""

        return LearningSheet(
            title=f"Learning Sheet: {topic}",
            content=content
        )


def generate_learning_sheet_from_snippets(topic: str, formatted_search_results: str) -> LearningSheet:
    """
    Generate learning sheet using search snippets (current implementation).

    This is the main entry point that maintains compatibility with the existing codebase
    while using the new configurable backend system.
    """
    if settings.logging.verbose_output:
        backend_type = settings.agent.backend_type
        print(f"📝 Generating learning sheet using {backend_type} backend")

    # Create agent instance
    generator = SheetGeneratorAgent()

    return generator.generate_from_snippets(topic, formatted_search_results)


def generate_learning_sheet_from_content(topic: str, scraped_contents: List[ScrapedContent]) -> EnhancedLearningSheet:
    """
    Generate enhanced learning sheet using full scraped content.

    TODO: Implement in Phase 4
    - Use full content instead of snippets
    - Generate enhanced output with metadata
    - Include proper citations and confidence scores
    """
    # Create agent instance
    generator = SheetGeneratorAgent()

    return generator.generate_from_content(topic, scraped_contents)
