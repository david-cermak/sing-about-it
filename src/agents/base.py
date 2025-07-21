"""
Base classes and interfaces for agent backends.

This module provides the foundation for supporting multiple agent backends:
1. Pydantic.ai backend (current implementation)
2. Baremetal OpenAI API backend (simple wrapper)

The interface remains the same regardless of backend choice.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar, Union
from pydantic import BaseModel
import json

# Type variable for Pydantic model return types
T = TypeVar('T', bound=BaseModel)


class AgentBackend(ABC):
    """
    Abstract base class for agent backends.

    This defines the interface that all agent backends must implement,
    allowing seamless switching between pydantic.ai and baremetal approaches.
    """

    def __init__(self, model_name: str, base_url: str, system_prompt: str):
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = system_prompt

    @abstractmethod
    def generate_response(self, prompt: str, output_type: Type[T]) -> T:
        """
        Generate a structured response from the LLM.

        Args:
            prompt: The user prompt to send to the LLM
            output_type: Pydantic model class defining expected output structure

        Returns:
            Instance of output_type with the LLM's structured response
        """
        pass


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class ValidationError(AgentError):
    """Raised when LLM output doesn't match expected structure."""
    pass


class APIError(AgentError):
    """Raised when there are issues with the underlying API calls."""
    pass


def validate_and_create_model(raw_response: str, output_type: Type[T]) -> T:
    """
    Validate raw LLM response and create Pydantic model instance.

    This utility function handles the common task of parsing LLM JSON responses
    and creating validated Pydantic model instances, with robust error handling.

    Args:
        raw_response: Raw string response from LLM
        output_type: Pydantic model class to create

    Returns:
        Validated instance of output_type

    Raises:
        ValidationError: If response doesn't match expected structure
    """
    try:
        # Clean common LLM response artifacts
        cleaned_response = _clean_llm_json_response(raw_response)

        # Parse JSON
        parsed_data = json.loads(cleaned_response)

        # Create and validate Pydantic model
        return output_type(**parsed_data)

    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in LLM response: {e}")
    except Exception as e:
        raise ValidationError(f"Failed to create {output_type.__name__} from response: {e}")


def _clean_llm_json_response(raw_response: str) -> str:
    """
    Clean common LLM JSON response formatting issues.

    LLMs often return JSON wrapped in markdown code blocks or with
    additional explanatory text. This function extracts clean JSON.
    """
    # Remove markdown code block formatting
    if "```json" in raw_response:
        raw_response = raw_response.split("```json")[1].split("```")[0]
    elif "```" in raw_response:
        raw_response = raw_response.split("```")[1].split("```")[0]

    # Remove common prefixes
    prefixes_to_remove = [
        "Here's the evaluation:",
        "Here is the evaluation:",
        "Evaluation:",
        "JSON:",
        "Response:",
        "Here's my analysis:",
        "Analysis:",
    ]

    for prefix in prefixes_to_remove:
        if raw_response.strip().lower().startswith(prefix.lower()):
            raw_response = raw_response[len(prefix):].strip()

    # Find JSON object bounds
    start_idx = raw_response.find('{')
    end_idx = raw_response.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        raw_response = raw_response[start_idx:end_idx+1]

    return raw_response.strip()
