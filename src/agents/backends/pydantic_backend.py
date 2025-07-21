"""
Pydantic.ai backend implementation.

This backend wraps pydantic.ai functionality to conform to the
common agent interface, allowing it to be used interchangeably
with other backends.
"""

from typing import Type, TypeVar
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.base import AgentBackend, APIError

T = TypeVar('T', bound=BaseModel)


class PydanticAIBackend(AgentBackend):
    """
    Backend implementation using pydantic.ai.

    This provides structured output validation, automatic retries,
    and rich debugging capabilities through the pydantic.ai framework.
    """

    def __init__(self, model_name: str, base_url: str, system_prompt: str):
        super().__init__(model_name, base_url, system_prompt)

        # Configure logfire for debugging if available
        try:
            import logfire
            logfire.configure(send_to_logfire=False)
            logfire.instrument_pydantic_ai()
        except ImportError:
            pass  # Logfire not available, continue without it

        # Create the OpenAI model instance
        self.model = OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url=base_url)
        )

    def generate_response(self, prompt: str, output_type: Type[T]) -> T:
        """
        Generate structured response using pydantic.ai.

        Args:
            prompt: User prompt for the LLM
            output_type: Pydantic model class for structured output

        Returns:
            Validated instance of output_type

        Raises:
            APIError: If the pydantic.ai agent fails to generate valid output
        """
        try:
            # Create agent with the specific output type
            agent = Agent(
                self.model,
                output_type=output_type,
                system_prompt=self.system_prompt
            )

            # Generate response
            result = agent.run_sync(prompt)
            return result.data

        except Exception as e:
            raise APIError(f"Pydantic.ai backend failed: {e}")
