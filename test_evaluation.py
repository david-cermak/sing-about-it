"""
Standalone test script to diagnose source evaluation issues
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_agent():
    """Test basic agent functionality without complex models."""
    print("🧪 Testing basic agent functionality...")

    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic import BaseModel, Field

        # Simple test model
        class SimpleResponse(BaseModel):
            score: float = Field(description="A simple score from 0 to 1")
            message: str = Field(description="A simple message")

        # Create basic agent
        model = OpenAIModel(
            model_name="llama3.2",
            provider=OpenAIProvider(base_url="http://127.0.0.1:11434/v1")
        )

        agent = Agent(
            model,
            output_type=SimpleResponse,
            system_prompt="You are a test agent. Return a JSON object with 'score' (0.0-1.0) and 'message' (string). Return ONLY valid JSON."
        )

        print("📤 Testing simple agent...")
        result = agent.run_sync("Give me a test response with score 0.8 and message 'hello'")
        print(f"✅ Success! Result: {result.data}")
        return True

    except Exception as e:
        print(f"❌ Basic agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_source_evaluation_direct():
    """Test source evaluation with simplified approach."""
    print("\n🧪 Testing source evaluation directly...")

    try:
        from config.models import SourceEvaluation
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider

        # Create simplified agent
        model = OpenAIModel(
            model_name="llama3.2",
            provider=OpenAIProvider(base_url="http://127.0.0.1:11434/v1")
        )

        simple_prompt = """You are evaluating a web source. Return ONLY a JSON object with these exact fields:

{
  "url": "the-url-here",
  "title": "the-title-here",
  "relevance_score": 0.8,
  "authority_score": 0.7,
  "content_type": "blog",
  "should_scrape": true,
  "reasoning": "This looks relevant",
  "estimated_quality": "medium"
}

Replace the values appropriately. Return ONLY the JSON, no other text."""

        agent = Agent(
            model,
            output_type=SourceEvaluation,
            system_prompt=simple_prompt
        )

        test_prompt = """Evaluate this source:
URL: https://example.com/ml-guide
Title: Machine Learning Basics Guide
Snippet: A comprehensive guide to machine learning fundamentals
Topic: machine learning"""

        print("📤 Testing source evaluation...")
        result = agent.run_sync(test_prompt)
        print(f"✅ Success! Result: {result.data}")
        return True

    except Exception as e:
        print(f"❌ Source evaluation test failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def test_raw_llm_response():
    """Test what the LLM actually returns without pydantic validation."""
    print("\n🧪 Testing raw LLM response...")

    try:
        import requests
        import json

        # Test direct API call to see raw response
        url = "http://127.0.0.1:11434/v1/chat/completions"

        payload = {
            "model": "llama3.2",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a test assistant. Return ONLY valid JSON: {\"test\": \"success\", \"score\": 0.5}"
                },
                {
                    "role": "user",
                    "content": "Give me the test JSON"
                }
            ],
            "temperature": 0.1
        }

        print("📤 Making direct API call...")
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ Raw response: {content}")

            # Try to parse as JSON
            try:
                parsed = json.loads(content)
                print(f"✅ Successfully parsed JSON: {parsed}")
                return True
            except json.JSONDecodeError as je:
                print(f"❌ JSON parsing failed: {je}")
                print(f"❌ Raw content was: '{content}'")
                return False
        else:
            print(f"❌ API call failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ Raw LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔍 Diagnosing Source Evaluation Issues")
    print("=" * 50)

    # Test 1: Basic agent functionality
    basic_success = test_basic_agent()

    # Test 2: Source evaluation with our model
    if basic_success:
        eval_success = test_source_evaluation_direct()
    else:
        eval_success = False

    # Test 3: Raw LLM response
    raw_success = test_raw_llm_response()

    print("\n" + "=" * 50)
    print("🔍 DIAGNOSIS SUMMARY:")
    print(f"  Basic Agent: {'✅' if basic_success else '❌'}")
    print(f"  Source Eval: {'✅' if eval_success else '❌'}")
    print(f"  Raw LLM:    {'✅' if raw_success else '❌'}")

    if not basic_success:
        print("\n💡 RECOMMENDATION: Check Ollama connection and llama3.2 model")
    elif not eval_success:
        print("\n💡 RECOMMENDATION: Issue with SourceEvaluation model or prompt format")
    elif not raw_success:
        print("\n💡 RECOMMENDATION: LLM is not returning valid JSON")
    else:
        print("\n✅ All tests passed! The issue might be elsewhere.")
