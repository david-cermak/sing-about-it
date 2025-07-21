"""
Baremetal OpenAI API backend implementation.

This backend provides direct OpenAI API access with manual JSON parsing,
following the pattern from the user's agent.py file. It offers more control
and simpler debugging while maintaining the same interface.
"""

import os
import time
import random
import json
import requests
from typing import Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.base import AgentBackend, APIError, ValidationError, validate_and_create_model

T = TypeVar('T', bound=BaseModel)


class BaremetalBackend(AgentBackend):
    """
    Backend implementation using direct OpenAI API calls.

    This provides a simple, transparent approach to LLM interaction
    with manual JSON parsing and validation. Based on the user's
    agent.py pattern with retry logic and exponential backoff.
    """

    def __init__(self, model_name: str, base_url: str, system_prompt: str, api_key: str = None):
        super().__init__(model_name, base_url, system_prompt)

        # Initialize OpenAI client (for compatibility)
        self.client = OpenAI(
            api_key=api_key or os.environ.get("API_KEY", ""),
            base_url=base_url
        )

        # Extract base URL for direct Ollama calls
        if "/v1" in base_url:
            self.ollama_base_url = base_url.replace("/v1", "")
        else:
            self.ollama_base_url = base_url

    def generate_response(self, prompt: str, output_type: Type[T]) -> T:
        """
        Generate structured response using direct API calls.

        Args:
            prompt: User prompt for the LLM
            output_type: Pydantic model class for structured output

        Returns:
            Validated instance of output_type

        Raises:
            APIError: If API calls fail after retries
            ValidationError: If response doesn't match expected structure
        """
        # Create enhanced system prompt with JSON structure requirements
        enhanced_system_prompt = self._create_json_system_prompt(output_type)

        # Try multiple strategies for best compatibility
        strategies = [
            self._try_ollama_direct,
            self._try_openai_compatible,
        ]

        for strategy in strategies:
            try:
                raw_response = strategy(prompt, enhanced_system_prompt)
                if raw_response:
                    return validate_and_create_model(raw_response, output_type)
            except Exception as e:
                continue  # Try next strategy

        raise APIError("All API strategies failed to generate valid response")

    def _try_ollama_direct(self, prompt: str, system_prompt: str) -> str:
        """Try direct Ollama API call."""
        url = f"{self.ollama_base_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        response = self._make_request_with_retry(url, payload)
        return response.json()["response"]

    def _try_openai_compatible(self, prompt: str, system_prompt: str) -> str:
        """Try OpenAI-compatible API call."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Try with requests first (more reliable for local setups)
        try:
            url = f"{self.client.base_url}chat/completions"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 4000
            }

            headers = {"Content-Type": "application/json"}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"

            response = self._make_request_with_retry(url, payload, headers)
            return response.json()["choices"][0]["message"]["content"]

        except Exception:
            # Fallback to OpenAI client
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content

    def _make_request_with_retry(self, url: str, payload: dict, headers: dict = None) -> requests.Response:
        """
        Make HTTP request with exponential backoff retry logic.

        Based on the retry pattern from the user's agent.py file.
        """
        max_retries = 5
        retry_count = 0
        base_delay = 5

        while retry_count < max_retries:
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers or {"Content-Type": "application/json"},
                    timeout=30
                )

                if response.status_code == 200:
                    return response
                else:
                    raise requests.RequestException(f"HTTP {response.status_code}: {response.text}")

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise APIError(f"Failed after {max_retries} attempts. Error: {str(e)}")

                # Calculate exponential backoff with jitter
                delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                time.sleep(delay)

    def _create_json_system_prompt(self, output_type: Type[T]) -> str:
        """
        Create enhanced system prompt with JSON schema requirements.

        This ensures the LLM understands exactly what JSON structure to return.
        """
        # Get field information from Pydantic model
        schema = output_type.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Create example JSON structure
        example_json = {}
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            if field_type == "boolean":
                example_json[field_name] = True
            elif field_type == "number":
                example_json[field_name] = 0.8
            elif field_type == "integer":
                example_json[field_name] = 42
            else:
                example_json[field_name] = "example_value"

        example_json_str = json.dumps(example_json, indent=2)

        return f"""{self.system_prompt}

CRITICAL: You MUST return ONLY a valid JSON object with these exact fields:

Required fields: {required}
All fields: {list(properties.keys())}

Example format:
{example_json_str}

IMPORTANT:
1. Return ONLY the JSON object, no other text
2. Do not wrap in markdown code blocks
3. Use proper JSON syntax with quotes around strings
4. Boolean values should be true/false (not "true"/"false")
5. Numbers should be numeric values, not strings

Return your response as valid JSON only."""
