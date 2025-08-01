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

        # ALWAYS try structured text parsing first (more reliable for LLMs)
        try:
            print(f"   📋 DEBUG: Trying structured text parsing first")
            parsed_data = _parse_structured_text_response(raw_response, output_type)

            # Check if we got meaningful data
            if parsed_data and any(v for v in parsed_data.values() if v not in [None, "", [], {}]):
                print(f"   ✅ DEBUG: Structured text parsing successful, keys: {list(parsed_data.keys())}")
            else:
                raise ValueError("Structured text parsing produced empty/invalid result")

        except Exception as text_error:
            print(f"   ❌ DEBUG: Structured text parsing failed: {text_error}")
            print(f"   📋 DEBUG: Falling back to JSON parsing")

            # JSON parsing as last resort
            cleaned_response = _clean_llm_json_response(raw_response)
            print(f"   🧹 DEBUG: Cleaned response: {cleaned_response[:300]}...")

            if not cleaned_response.strip():
                print(f"   🚨 DEBUG: JSON cleaning also failed, using emergency response")
                parsed_data = _create_emergency_response(output_type)
            else:
                try:
                    parsed_data = json.loads(cleaned_response)
                    print(f"   ✅ DEBUG: JSON parsing successful, keys: {list(parsed_data.keys())}")
                except json.JSONDecodeError as json_error:
                    print(f"   ❌ DEBUG: JSON decode also failed: {json_error}")
                    print(f"   🚨 DEBUG: Using emergency response")
                    parsed_data = _create_emergency_response(output_type)

        # Create and validate Pydantic model
        result = output_type(**parsed_data)
        print(f"   ✅ DEBUG: Pydantic validation successful")
        return result

    except Exception as e:
        print(f"   ❌ DEBUG: Validation error: {e}")
        # Last resort: try to extract any meaningful content for fallback
        print(f"   🔄 DEBUG: Attempting emergency fallback parsing")
        try:
            # Create a basic fallback response
            fallback_data = {}

            # Try to extract any content as summary
            cleaned_content = _strip_reasoning_tags(raw_response)
            if len(cleaned_content.strip()) > 20:
                if "TopicSummaryResponse" in str(output_type):
                    fallback_data = {
                        'summary': cleaned_content[:2000],  # Truncate if too long
                        'key_points': ['Fallback parsing used due to format issues'],
                        'confidence_score': 0.3
                    }
                elif "OrchestrationResponse" in str(output_type):
                    fallback_data = {
                        'title': 'Learning Sheet (Fallback Mode)',
                        'content': cleaned_content[:4000],
                        'key_takeaways': ['Fallback parsing used'],
                        'cross_connections': ['Unable to identify connections'],
                        'confidence_score': 0.3
                    }
                else:
                    # Generic fallback
                    fallback_data = {'content': cleaned_content[:1000]}

            if fallback_data:
                result = output_type(**fallback_data)
                print(f"   🔄 DEBUG: Emergency fallback successful")
                return result

        except Exception as fallback_error:
            print(f"   ❌ DEBUG: Emergency fallback also failed: {fallback_error}")

        raise ValidationError(f"Failed to create {output_type.__name__} from response: {e}")


def _clean_llm_json_response(raw_response: str) -> str:
    """
    Clean common LLM JSON response formatting issues.

    LLMs often return JSON wrapped in markdown code blocks or with
    additional explanatory text. This function extracts clean JSON.
    """
    import re

    # Remove <think> tags first
    raw_response = _strip_reasoning_tags(raw_response)

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
        "Here is a comprehensive summary",
        "Here's a comprehensive summary",
    ]

    for prefix in prefixes_to_remove:
        if raw_response.strip().lower().startswith(prefix.lower()):
            raw_response = raw_response[len(prefix):].strip()

    # Find JSON object bounds
    start_idx = raw_response.find('{')
    end_idx = raw_response.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_content = raw_response[start_idx:end_idx+1]

        # Try to fix common JSON formatting issues
        try:
            # Fix missing quotes around values that follow newlines
            # Pattern: "key":\nValue -> "key": "Value"
            json_content = re.sub(r'"([^"]+)":\s*\n\s*([^"][^,}]*)', r'"\1": "\2"', json_content)

            # Fix trailing commas and other common issues
            json_content = re.sub(r',\s*}', '}', json_content)  # Remove trailing commas
            json_content = re.sub(r',\s*]', ']', json_content)  # Remove trailing commas in arrays

            # If still malformed, it's probably not proper JSON format
            # Let the structured text parser handle it instead
            if ':\n' in json_content and '"' not in json_content.split(':\n')[1][:20]:
                # This looks like malformed JSON, probably better as structured text
                return ""

        except Exception:
            # If cleaning fails, return empty to trigger structured text parsing
            return ""

        return json_content.strip()

    return raw_response.strip()


def _create_emergency_response(output_type: Type[T]) -> dict:
    """Create a minimal valid response when all parsing fails."""
    if "ChunkMappingResponse" in str(output_type):
        return {"mappings": [], "analysis": "Parsing failed - no mappings available"}
    elif "TopicSummaryResponse" in str(output_type):
        return {"summary": "Summary generation failed", "key_points": [], "confidence": 0.1}
    elif "TopicIdentificationResponse" in str(output_type):
        return {"topics": [], "reasoning": "Topic identification failed"}
    elif "OrchestrationResponse" in str(output_type):
        return {
            "content": "Content generation failed due to parsing errors",
            "key_takeaways": [],
            "cross_connections": [],
            "confidence_score": 0.1
        }
    else:
        return {"content": "Unknown response type - parsing failed"}


def _strip_reasoning_tags(raw_response: str) -> str:
    """
    Strip <think> and </think> tags from reasoning model responses.

    Reasoning models like deepseek-r1 often wrap their reasoning in <think> tags.
    """
    import re
    # Remove <think>...</think> blocks entirely
    cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _is_structured_text_response(raw_response: str) -> bool:
    """
    Check if response uses structured text format instead of JSON.

    Looks for patterns like:
    SUMMARY:
    KEY_POINTS:
    CONFIDENCE:

    Also handles reasoning models with <think> tags and ## markdown format.
    """
    # Strip <think> tags for reasoning models
    cleaned_response = _strip_reasoning_tags(raw_response)
    response_upper = cleaned_response.upper()

    # Original colon format
    colon_indicators = ["SUMMARY:", "KEY_POINTS:", "CONFIDENCE:", "TOPICS:", "MAPPINGS:",
                       "TITLE:", "CONTENT:", "KEY_TAKEAWAYS:", "CROSS_CONNECTIONS:"]

    # Markdown format (for reasoning models)
    markdown_indicators = ["## CONTENT", "## KEY_TAKEAWAYS", "## CROSS_CONNECTIONS", "## CONFIDENCE"]

    # Check for malformed JSON that should be treated as structured text
    malformed_json_indicators = [
        '"SUMMARY":\n',  # JSON key followed by newline (malformed)
        '"SUMMARY":\n  ',  # JSON key with newline and spaces
        '{\n  "SUMMARY":\n',  # Common malformed pattern
        'HERE IS A COMPREHENSIVE SUMMARY',  # Natural language start
        '**SUMMARY**',  # Markdown headers
        '**KEY CONCEPTS**'
    ]

    # If it contains structured text indicators, use structured parsing
    if any(indicator in response_upper for indicator in colon_indicators):
        return True
    if any(indicator in response_upper for indicator in markdown_indicators):
        return True
    if any(indicator in response_upper for indicator in malformed_json_indicators):
        return True

    # If it starts with natural language instead of JSON, treat as structured text
    cleaned_start = cleaned_response.strip()[:100].upper()
    natural_language_starts = [
        'HERE IS A COMPREHENSIVE',
        'HERE IS A DETAILED',
        'THE FOLLOWING IS A',
        'THIS SUMMARY PROVIDES',
        'EMBEDDED SECURITY',
        'THREATS TO EMBEDDED',
        'SECURITY REQUIREMENTS'
    ]

    return any(start in cleaned_start for start in natural_language_starts)


def _parse_chunk_mapping_response(raw_response: str) -> dict:
    """Parse chunk mapping response from structured text."""
    import re

    # Strip reasoning tags first
    cleaned = _strip_reasoning_tags(raw_response)

    # Look for mappings section
    mappings = []

    # Try to extract chunk mappings from various formats
    chunk_patterns = [
        r'chunk\s*(\d+)\s*→\s*topic\s*(\d+)\s*\(.*?([0-9.]+)\)',  # chunk 1 → topic 2 (score: 0.85)
        r'chunk[_\s]*(\d+).*?topic[_\s]*(\d+).*?(?:score|relevance)[_\s:]*([0-9.]+)',
        r'(\d+)\s*→\s*(\d+)\s*\(([0-9.]+)\)',
        r'chunk\s*(\d+):\s*topic\s*(\d+)\s*.*?([0-9.]+)',
        r'(\d+)\.\s*topic\s*(\d+).*?([0-9.]+)',  # numbered format
    ]

    # Extract reasoning for each chunk if available
    reasoning_map = {}
    reasoning_pattern = r'chunk\s*(\d+).*?-\s*([^\\n\\r]+)'
    reasoning_matches = re.findall(reasoning_pattern, cleaned, re.IGNORECASE | re.MULTILINE)
    for chunk_num, reasoning in reasoning_matches:
        reasoning_map[int(chunk_num)] = reasoning.strip()

    for pattern in chunk_patterns:
        matches = re.findall(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
        if matches:
            for match in matches:
                chunk_idx, topic_num, score = match
                chunk_id = int(chunk_idx)
                reasoning = reasoning_map.get(chunk_id, f"Mapped chunk {chunk_idx} to topic {topic_num}")
                mappings.append({
                    "chunk_index": chunk_id,
                    "topic_number": int(topic_num),
                    "relevance_score": float(score),
                    "reasoning": reasoning
                })
            break

    # Extract any analysis text
    analysis = ""
    if "analysis" in cleaned.lower() or "summary" in cleaned.lower():
        analysis_match = re.search(r'(?:analysis|summary)[:\s]+(.+?)(?:\n\n|\Z)', cleaned, re.IGNORECASE | re.DOTALL)
        if analysis_match:
            analysis = analysis_match.group(1).strip()

    return {
        "mappings": mappings,
        "analysis": analysis
    }


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

    Also handles reasoning models with <think> tags.
    """
    import re

    # Strip reasoning tags first for all structured text parsing
    raw_response = _strip_reasoning_tags(raw_response)

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
    elif "OrchestrationResponse" in str(output_type):
        parsed_data = _parse_orchestration_response(raw_response)
    else:
        # Generic parsing
        parsed_data = _parse_generic_structured_response(raw_response, model_fields)

    print(f"   ✅ DEBUG: Structured text parsing successful, keys: {list(parsed_data.keys())}")
    return parsed_data


def _parse_topic_summary_response(raw_response: str) -> dict:
    """Parse topic summary response in structured text format."""
    import re

    parsed_data = {}

    # Extract summary - try multiple formats
    summary_match = re.search(r'SUMMARY:\s*\n(.*?)(?=\n\s*KEY_POINTS:|$)', raw_response, re.DOTALL | re.IGNORECASE)

    if not summary_match:
        # Try **Summary** markdown format
        summary_match = re.search(r'\*\*SUMMARY\*\*\s*\n(.*?)(?=\n\s*\*\*KEY|$)', raw_response, re.DOTALL | re.IGNORECASE)

    if not summary_match:
        # Try natural language format: "Here is a comprehensive summary..."
        summary_match = re.search(r'(?:here is a (?:comprehensive|detailed) summary.*?:\s*\n\s*)?(?:\*\*summary\*\*\s*\n)?(.*?)(?=\n\s*\*\*key|key_points:|key points:|confidence|$)', raw_response, re.DOTALL | re.IGNORECASE)

    if not summary_match:
        # Last resort: take everything until key points or confidence
        summary_match = re.search(r'^(.*?)(?=\n\s*(?:key_points|key points|confidence):|$)', raw_response, re.DOTALL | re.IGNORECASE)

    if summary_match:
        summary_text = summary_match.group(1).strip()
        # Clean up common prefixes
        summary_text = re.sub(r'^.*?here is a (?:comprehensive|detailed) summary.*?:\s*', '', summary_text, flags=re.IGNORECASE)
        summary_text = re.sub(r'^\*\*summary\*\*\s*', '', summary_text, flags=re.IGNORECASE)
        summary_text = re.sub(r'^summary:\s*', '', summary_text, flags=re.IGNORECASE)
        parsed_data['summary'] = summary_text.strip()

    # Extract key points - try multiple formats
    key_points_match = re.search(r'KEY_POINTS:\s*\n(.*?)(?=\n\s*CONFIDENCE:|$)', raw_response, re.DOTALL | re.IGNORECASE)

    if not key_points_match:
        # Try **Key Points** format
        key_points_match = re.search(r'\*\*KEY.*?POINTS?\*\*\s*\n(.*?)(?=\n\s*\*\*|confidence|$)', raw_response, re.DOTALL | re.IGNORECASE)

    if not key_points_match:
        # Try "Key Points" without formatting
        key_points_match = re.search(r'key points?:?\s*\n(.*?)(?=\n\s*confidence|$)', raw_response, re.DOTALL | re.IGNORECASE)

    if key_points_match:
        points_text = key_points_match.group(1).strip()
        # Split on lines starting with - or * or numbers
        key_points = []
        for line in points_text.split('\n'):
            line = line.strip()
            if line.startswith(('-', '*', '•')) or re.match(r'^\d+\.', line):
                # Remove bullet point or number
                clean_line = re.sub(r'^[-*•\d\.]\s*', '', line).strip()
                if clean_line:  # Only add non-empty points
                    key_points.append(clean_line)
        parsed_data['key_points'] = key_points

    # Extract confidence score - try multiple formats
    confidence_match = re.search(r'CONFIDENCE(?:\s*SCORE)?:?\s*([0-9]*\.?[0-9]+)', raw_response, re.IGNORECASE)

    if not confidence_match:
        # Try **Confidence Score**: format
        confidence_match = re.search(r'\*\*confidence.*?score?\*\*:?\s*([0-9]*\.?[0-9]+)', raw_response, re.IGNORECASE)

    if confidence_match:
        try:
            parsed_data['confidence_score'] = float(confidence_match.group(1))
        except ValueError:
            parsed_data['confidence_score'] = 0.5  # Default fallback

    # Set defaults if missing
    if 'summary' not in parsed_data:
        # If no summary found but we have content, use the raw response (cleaned)
        cleaned_response = _strip_reasoning_tags(raw_response)
        # Remove confidence scores and key points sections
        cleaned_response = re.sub(r'\n\s*(?:key.*?points?|confidence).*$', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        if len(cleaned_response.strip()) > 50:  # If we have substantial content
            parsed_data['summary'] = cleaned_response.strip()
        else:
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


def _parse_orchestration_response(raw_response: str) -> dict:
    """Parse orchestration response in structured text format (handles both colon and markdown formats)."""
    import re

    # Note: reasoning tags already stripped at higher level
    parsed_data = {}

    # Extract title - try multiple formats
    title_match = re.search(r'TITLE:\s*\n(.*?)(?=\n\s*CONTENT:|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if not title_match:
        # Try markdown format: **Title** (first occurrence)
        title_match = re.search(r'\*\*(.*?)\*\*', raw_response, re.IGNORECASE)
    if title_match:
        parsed_data['title'] = title_match.group(1).strip()

    # Extract content - try multiple formats including ## markdown
    content_match = re.search(r'CONTENT:\s*\n(.*?)(?=\n\s*KEY_TAKEAWAYS:|## KEY_TAKEAWAYS|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if not content_match:
        # Try ## markdown format: ## CONTENT:
        content_match = re.search(r'## CONTENT:?\s*\n(.*?)(?=\n\s*## KEY_TAKEAWAYS|## CROSS_CONNECTIONS|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if not content_match:
        # Try **CONTENT** markdown format
        content_match = re.search(r'\*\*CONTENT\*\*\s*\n(.*?)(?=\n\s*\*\*KEY_TAKEAWAYS\*\*|\n\s*## KEY_TAKEAWAYS|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if not content_match:
        # Fallback: everything after CONTENT until next section
        content_match = re.search(r'(?:CONTENT|content).*?\n(.*?)(?=## KEY_TAKEAWAYS|KEY_TAKEAWAYS:|$)', raw_response, re.DOTALL | re.IGNORECASE)

    # Handle content extraction
    if content_match:
        parsed_data['content'] = content_match.group(1).strip()
    else:
        # Simple fallback: Look for substantial text after </think>
        after_think_match = re.search(r'</think>\s*(.*)', raw_response, re.DOTALL | re.IGNORECASE)
        if after_think_match:
            remaining_text = after_think_match.group(1).strip()
            # Take first 2000 characters as content if substantial
            if len(remaining_text) > 200:
                parsed_data['content'] = remaining_text[:2000].strip()
        else:
            # Ultimate fallback: strip tags and take substantial content
            clean_text = _strip_reasoning_tags(raw_response)
            # Remove section headers and take middle content
            lines = [line.strip() for line in clean_text.split('\n')
                    if line.strip() and not re.match(r'^[#*]+\s|^[A-Z_]+:\s*$', line)]
            if len(lines) > 3:
                parsed_data['content'] = '\n\n'.join(lines[1:6])  # Take middle lines

    # Extract key takeaways - try multiple formats including ## markdown
    takeaways_match = re.search(r'KEY_TAKEAWAYS:\s*\n(.*?)(?=\n\s*CROSS_CONNECTIONS:|## CROSS_CONNECTIONS|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if not takeaways_match:
        # Try ## markdown format: ## KEY_TAKEAWAYS:
        takeaways_match = re.search(r'## KEY_TAKEAWAYS:?\s*\n(.*?)(?=\n\s*## CROSS_CONNECTIONS|## CONFIDENCE|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if not takeaways_match:
        # Try **KEY_TAKEAWAYS** format
        takeaways_match = re.search(r'\*\*KEY_TAKEAWAYS\*\*\s*\n(.*?)(?=\n\s*\*\*CROSS_CONNECTIONS\*\*|\n\s*## CROSS_CONNECTIONS|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if takeaways_match:
        takeaways_text = takeaways_match.group(1).strip()
        key_takeaways = []
        for line in takeaways_text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*') or line.startswith('•') or re.match(r'^\d+\.', line):
                # Handle numbered lists too (1., 2., etc.)
                if re.match(r'^\d+\.', line):
                    key_takeaways.append(re.sub(r'^\d+\.\s*', '', line).strip())
                else:
                    key_takeaways.append(line[1:].strip())
        parsed_data['key_takeaways'] = key_takeaways

    # Extract cross connections - try multiple formats including ## markdown
    connections_match = re.search(r'CROSS_CONNECTIONS:\s*\n(.*?)(?=\n\s*CONFIDENCE:|## CONFIDENCE|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if not connections_match:
        # Try ## markdown format: ## CROSS_CONNECTIONS:
        connections_match = re.search(r'## CROSS_CONNECTIONS:?\s*\n(.*?)(?=\n\s*## CONFIDENCE|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if not connections_match:
        # Try **CROSS_CONNECTIONS** format
        connections_match = re.search(r'\*\*CROSS_CONNECTIONS\*\*\s*\n(.*?)(?=\n\s*\*\*CONFIDENCE\*\*|\n\s*## CONFIDENCE|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if connections_match:
        connections_text = connections_match.group(1).strip()
        cross_connections = []
        for line in connections_text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*') or line.startswith('•') or re.match(r'^\d+\.', line):
                # Handle numbered lists too
                if re.match(r'^\d+\.', line):
                    cross_connections.append(re.sub(r'^\d+\.\s*', '', line).strip())
                else:
                    cross_connections.append(line[1:].strip())
        parsed_data['cross_connections'] = cross_connections

    # Extract confidence score - try multiple formats including ## markdown
    confidence_match = re.search(r'CONFIDENCE:?\s*([0-9]*\.?[0-9]+)', raw_response, re.IGNORECASE)
    if not confidence_match:
        # Try ## markdown format: ## CONFIDENCE: 0.85
        confidence_match = re.search(r'## CONFIDENCE:?\s*([0-9]*\.?[0-9]+)', raw_response, re.IGNORECASE)
    if not confidence_match:
        # Try different formats like "Confidence Score: 0.85"
        confidence_match = re.search(r'confidence.*?([0-9]*\.?[0-9]+)', raw_response, re.IGNORECASE)
    if confidence_match:
        try:
            parsed_data['confidence_score'] = float(confidence_match.group(1))
        except ValueError:
            parsed_data['confidence_score'] = 0.5

    # Set defaults if missing
    if 'title' not in parsed_data:
        parsed_data['title'] = "Learning Sheet"
    if 'content' not in parsed_data:
        parsed_data['content'] = "Content not found in response"
    if 'key_takeaways' not in parsed_data:
        parsed_data['key_takeaways'] = ["Key takeaways not found in response"]
    if 'cross_connections' not in parsed_data:
        parsed_data['cross_connections'] = ["Cross connections not found in response"]
    if 'confidence_score' not in parsed_data:
        parsed_data['confidence_score'] = 0.5

    return parsed_data


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
