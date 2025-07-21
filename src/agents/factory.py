"""
Agent factory for creating backend instances based on configuration.

This factory pattern allows seamless switching between different
agent backends while maintaining a consistent interface.
"""

from typing import Type, TypeVar
from enum import Enum
from pydantic import BaseModel
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.base import AgentBackend
from agents.backends.pydantic_backend import PydanticAIBackend
from agents.backends.baremetal_backend import BaremetalBackend
from config.settings import settings

T = TypeVar('T', bound=BaseModel)


class BackendType(Enum):
    """Supported agent backend types."""
    PYDANTIC_AI = "pydantic_ai"
    BAREMETAL = "baremetal"


class AgentFactory:
    """
    Factory for creating agent backend instances.

    This factory chooses the appropriate backend implementation based on
    configuration settings while ensuring a consistent interface.
    """

    @staticmethod
    def create_backend(
        backend_type: BackendType,
        system_prompt: str,
        model_name: str = None,
        base_url: str = None
    ) -> AgentBackend:
        """
        Create an agent backend instance.

        Args:
            backend_type: Type of backend to create
            system_prompt: System prompt for the agent
            model_name: LLM model name (defaults to config)
            base_url: API base URL (defaults to config)

        Returns:
            Configured agent backend instance
        """
        # Use config defaults if not provided
        model_name = model_name or settings.llm.model
        base_url = base_url or settings.llm.base_url

        if backend_type == BackendType.PYDANTIC_AI:
            return PydanticAIBackend(
                model_name=model_name,
                base_url=base_url,
                system_prompt=system_prompt
            )
        elif backend_type == BackendType.BAREMETAL:
            return BaremetalBackend(
                model_name=model_name,
                base_url=base_url,
                system_prompt=system_prompt
            )
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

    @staticmethod
    def create_default_backend(system_prompt: str) -> AgentBackend:
        """
        Create backend using default configuration.

        The default backend type is determined by the AGENT_BACKEND
        environment variable or falls back to PYDANTIC_AI.
        """
        backend_type_str = settings.agent.backend_type.lower()

        if backend_type_str == "baremetal":
            backend_type = BackendType.BAREMETAL
        else:
            backend_type = BackendType.PYDANTIC_AI

        return AgentFactory.create_backend(backend_type, system_prompt)


# Convenience functions for common use cases
def create_source_evaluator_backend() -> AgentBackend:
    """Create backend for source evaluation agent."""
    system_prompt = """You are an expert source evaluation agent for educational content research.

Your task is to evaluate web sources for their quality, relevance, and value for creating comprehensive learning materials.

EVALUATION CRITERIA:

1. RELEVANCE SCORE (0.0-1.0):
   - 1.0: Directly addresses the topic with comprehensive coverage
   - 0.8: Highly relevant with good topic coverage
   - 0.6: Moderately relevant, covers some aspects
   - 0.4: Tangentially related or limited coverage
   - 0.2: Barely related to the topic
   - 0.0: Not related to the topic

2. AUTHORITY SCORE (0.0-1.0):
   - 1.0: Authoritative sources (gov, edu, established organizations, expert sites)
   - 0.8: Professional publications, established media, technical documentation
   - 0.6: Industry blogs, company websites, established platforms
   - 0.4: Personal blogs by experts, forums with good moderation
   - 0.2: General user-generated content, unverified sources
   - 0.0: Spam, promotional, or unreliable sources

3. CONTENT TYPE Classification:
   - "academic": Research papers, educational institutions, formal studies
   - "documentation": Official docs, technical guides, tutorials
   - "news": News articles, industry reports, current events
   - "blog": Expert blogs, thought leadership, analysis
   - "reference": Wikis, encyclopedias, fact sheets
   - "commercial": Company sites, product pages, marketing content

4. QUALITY ESTIMATION:
   - "high": Authoritative, comprehensive, well-structured content expected
   - "medium": Good information quality but may lack depth or authority
   - "low": Limited value, promotional, or superficial content expected

5. SCRAPING DECISION:
   - should_scrape: true if relevance ≥ 0.6 AND authority ≥ 0.3
   - Prioritize diverse content types and authoritative sources
   - Avoid duplicate domains when possible

6. REASONING:
   - Provide clear justification for scores and decision
   - Mention specific factors that influenced the evaluation
   - Note any red flags or positive indicators"""

    return AgentFactory.create_default_backend(system_prompt)


def create_sheet_generator_backend() -> AgentBackend:
    """Create backend for learning sheet generation agent."""
    system_prompt = (
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

    return AgentFactory.create_default_backend(system_prompt)
