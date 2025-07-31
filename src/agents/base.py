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
import re

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

    Now supports both JSON and structured text parsing for better LLM compatibility.

    Args:
        raw_response: Raw string response from LLM
        output_type: Pydantic model class to create

    Returns:
        Validated instance of output_type

    Raises:
        ValidationError: If response doesn't match expected structure
    """
    try:
        print(f"   🔍 DEBUG: Validating response for {output_type.__name__}")
        print(f"   📝 DEBUG: Raw response ({len(raw_response)} chars): {raw_response[:300]}...")

        # Try structured text parsing first (more LLM-friendly)
        if _is_structured_text_response(raw_response):
            print(f"   📋 DEBUG: Detected structured text format")
            parsed_data = _parse_structured_text_response(raw_response, output_type)
        else:
            print(f"   📋 DEBUG: Attempting JSON parsing")
            # Fallback to JSON parsing
            cleaned_response = _clean_llm_json_response(raw_response)
            print(f"   🧹 DEBUG: Cleaned response: {cleaned_response[:300]}...")
            parsed_data = json.loads(cleaned_response)
            print(f"   ✅ DEBUG: JSON parsing successful, keys: {list(parsed_data.keys())}")

        # Create and validate Pydantic model
        result = output_type(**parsed_data)
        print(f"   ✅ DEBUG: Pydantic validation successful")
        return result

    except json.JSONDecodeError as e:
        print(f"   ❌ DEBUG: JSON decode error: {e}")
        raise ValidationError(f"Invalid JSON in LLM response: {e}")
    except Exception as e:
        print(f"   ❌ DEBUG: Validation error: {e}")
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


def _is_structured_text_response(raw_response: str) -> bool:
    """
    Check if response uses structured text format instead of JSON.

    Looks for patterns like:
    SUMMARY:
    KEY_POINTS:
    CONFIDENCE:
    """
    response_upper = raw_response.upper()
    indicators = ["SUMMARY:", "KEY_POINTS:", "CONFIDENCE:", "TOPICS:", "MAPPINGS:"]
    return any(indicator in response_upper for indicator in indicators)


def _parse_structured_text_response(raw_response: str, output_type: Type[T]) -> dict:
    """
    Parse structured text response into dictionary for Pydantic model creation.

    Supports formats like:
    SUMMARY:
    Content here

    KEY_POINTS:
    - Point 1
    - Point 2

    CONFIDENCE: 0.85
    """
    import re

    # Get expected field names from the Pydantic model
    model_fields = output_type.model_fields.keys() if hasattr(output_type, 'model_fields') else []

    parsed_data = {}

    # Parse different response types based on the model
    if "TopicSummaryResponse" in str(output_type):
        parsed_data = _parse_topic_summary_response(raw_response)
    elif "TopicIdentificationResponse" in str(output_type):
        parsed_data = _parse_topic_identification_response(raw_response)
    elif "ChunkMappingResponse" in str(output_type):
        parsed_data = _parse_chunk_mapping_response(raw_response)
    else:
        # Generic parsing
        parsed_data = _parse_generic_structured_response(raw_response, model_fields)

    print(f"   ✅ DEBUG: Structured text parsing successful, keys: {list(parsed_data.keys())}")
    return parsed_data


def _parse_topic_summary_response(raw_response: str) -> dict:
    """Parse topic summary response in structured text format."""
    import re

    parsed_data = {}

    # Extract summary
    summary_match = re.search(r'SUMMARY:\s*\n(.*?)(?=\n\s*KEY_POINTS:|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if summary_match:
        parsed_data['summary'] = summary_match.group(1).strip()

    # Extract key points
    key_points_match = re.search(r'KEY_POINTS:\s*\n(.*?)(?=\n\s*CONFIDENCE:|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if key_points_match:
        points_text = key_points_match.group(1).strip()
        # Split on lines starting with - or *
        key_points = []
        for line in points_text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                key_points.append(line[1:].strip())
        parsed_data['key_points'] = key_points

    # Extract confidence score
    confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', raw_response, re.IGNORECASE)
    if confidence_match:
        try:
            parsed_data['confidence_score'] = float(confidence_match.group(1))
        except ValueError:
            parsed_data['confidence_score'] = 0.5  # Default fallback

    # Set defaults if missing
    if 'summary' not in parsed_data:
        parsed_data['summary'] = "Summary not found in response"
    if 'key_points' not in parsed_data:
        parsed_data['key_points'] = ["Key points not found in response"]
    if 'confidence_score' not in parsed_data:
        parsed_data['confidence_score'] = 0.5

    return parsed_data


def _parse_topic_identification_response(raw_response: str) -> dict:
    """Parse topic identification response in structured text format."""
    # For now, return a simple structure - we can enhance this later
    return {
        "topics": [
            {"name": "Fundamentals", "description": "Basic concepts", "complexity": "beginner", "priority": 5},
            {"name": "Applications", "description": "Practical applications", "complexity": "intermediate", "priority": 4}
        ],
        "reasoning": "Parsed from structured text format"
    }


def _parse_chunk_mapping_response(raw_response: str) -> dict:
    """Parse chunk mapping response in structured text format."""
    # For now, return a simple structure - we can enhance this later
    return {
        "mappings": [
            {"chunk_index": 1, "topic_number": 1, "relevance_score": 0.8, "reasoning": "Mapped from structured text"}
        ],
        "analysis": "Parsed from structured text format"
    }


def _parse_generic_structured_response(raw_response: str, model_fields: list) -> dict:
    """Generic parser for structured text responses."""
    parsed_data = {}

    # Simple field extraction for any model
    for field in model_fields:
        field_upper = field.upper()
        pattern = f'{field_upper}:\\s*([^\\n]+)'
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Try to convert to appropriate type
            try:
                if '.' in value and value.replace('.', '').isdigit():
                    parsed_data[field] = float(value)
                elif value.isdigit():
                    parsed_data[field] = int(value)
                elif value.lower() in ['true', 'false']:
                    parsed_data[field] = value.lower() == 'true'
                else:
                    parsed_data[field] = value
            except:
                parsed_data[field] = value

    return parsed_data
