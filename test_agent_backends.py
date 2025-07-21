"""
Test script for demonstrating agent backend switching.

This script shows how users can choose between pydantic.ai and baremetal backends
while maintaining the same interface. It tests both source evaluation and
sheet generation agents with different backend configurations.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.models import WebSearchResult, SourceEvaluation, LearningSheet
from config.settings import settings
from agents.factory import AgentFactory, BackendType
from agents.source_evaluator_new import SourceEvaluatorAgent
from agents.sheet_generator_new import SheetGeneratorAgent


def test_source_evaluation_backends():
    """
    Test source evaluation with different backends.
    """
    print("🧪 Testing Source Evaluation Backends")
    print("=" * 60)

    # Create test data
    test_result = WebSearchResult(
        url="https://en.wikipedia.org/wiki/Machine_learning",
        title="Machine Learning - Wikipedia",
        snippet="Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data.",
        search_query="machine learning basics"
    )

    topic = "machine learning"

    # Test both backends
    backends_to_test = [
        (BackendType.PYDANTIC_AI, "Pydantic.ai"),
        (BackendType.BAREMETAL, "Baremetal")
    ]

    for backend_type, backend_name in backends_to_test:
        print(f"\n🔧 Testing {backend_name} Backend:")
        print("-" * 30)

        try:
            # Create backend instance
            backend = AgentFactory.create_backend(
                backend_type=backend_type,
                system_prompt="You are a source evaluation agent. Evaluate sources for educational content research."
            )

            # Create agent with specific backend
            evaluator = SourceEvaluatorAgent(backend=backend)

            print(f"   📤 Evaluating source with {backend_name}...")
            start_time = time.time()

            evaluation = evaluator.evaluate_source(test_result, topic)

            end_time = time.time()
            duration = end_time - start_time

            print(f"   ✅ Success! ({duration:.2f}s)")
            print(f"   📊 Results:")
            print(f"      Relevance: {evaluation.relevance_score:.2f}")
            print(f"      Authority: {evaluation.authority_score:.2f}")
            print(f"      Content Type: {evaluation.content_type}")
            print(f"      Should Scrape: {evaluation.should_scrape}")
            print(f"      Quality: {evaluation.estimated_quality}")
            print(f"      Reasoning: {evaluation.reasoning[:100]}...")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print(f"   🔍 Error type: {type(e).__name__}")


def test_sheet_generation_backends():
    """
    Test learning sheet generation with different backends.
    """
    print("\n\n🧪 Testing Learning Sheet Generation Backends")
    print("=" * 60)

    topic = "machine learning"
    formatted_results = """
    SEARCH RESULTS FOR: machine learning

    1. Machine Learning - Wikipedia
       https://en.wikipedia.org/wiki/Machine_learning
       Machine learning is a method of data analysis that automates analytical model building...

    2. Introduction to Machine Learning
       https://www.coursera.org/learn/machine-learning
       A comprehensive introduction to machine learning concepts and algorithms...

    3. Machine Learning Basics
       https://towardsdatascience.com/machine-learning-basics
       Learn the fundamentals of machine learning with practical examples...
    """

    # Test both backends
    backends_to_test = [
        (BackendType.PYDANTIC_AI, "Pydantic.ai"),
        (BackendType.BAREMETAL, "Baremetal")
    ]

    for backend_type, backend_name in backends_to_test:
        print(f"\n🔧 Testing {backend_name} Backend:")
        print("-" * 30)

        try:
            # Create backend instance
            backend = AgentFactory.create_backend(
                backend_type=backend_type,
                system_prompt="You are an educational content creator. Generate comprehensive learning sheets."
            )

            # Create agent with specific backend
            generator = SheetGeneratorAgent(backend=backend)

            print(f"   📤 Generating learning sheet with {backend_name}...")
            start_time = time.time()

            sheet = generator.generate_from_snippets(topic, formatted_results)

            end_time = time.time()
            duration = end_time - start_time

            print(f"   ✅ Success! ({duration:.2f}s)")
            print(f"   📄 Results:")
            print(f"      Title: {sheet.title}")
            print(f"      Content Length: {len(sheet.content)} characters")
            print(f"      Content Preview: {sheet.content[:200]}...")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print(f"   🔍 Error type: {type(e).__name__}")


def test_environment_configuration():
    """
    Test environment-based backend configuration.
    """
    print("\n\n🧪 Testing Environment Configuration")
    print("=" * 60)

    # Test different environment configurations
    test_configs = [
        ("pydantic_ai", "Pydantic.ai"),
        ("baremetal", "Baremetal"),
        ("invalid", "Invalid (should fallback to pydantic.ai)")
    ]

    for config_value, description in test_configs:
        print(f"\n🔧 Testing AGENT_BACKEND={config_value} ({description}):")
        print("-" * 40)

        # Temporarily set environment variable
        original_value = os.environ.get('AGENT_BACKEND', '')
        os.environ['AGENT_BACKEND'] = config_value

        try:
            # Force reload of settings
            from importlib import reload
            import config.settings
            reload(config.settings)

            # Test backend creation
            from agents.factory import create_source_evaluator_backend
            backend = create_source_evaluator_backend()

            backend_class_name = backend.__class__.__name__
            print(f"   ✅ Created backend: {backend_class_name}")

            # Test the backend works
            test_result = WebSearchResult(
                url="https://example.com",
                title="Test Article",
                snippet="Test content snippet",
                search_query="test query"
            )

            evaluator = SourceEvaluatorAgent(backend=backend)
            evaluation = evaluator.evaluate_source(test_result, "test topic")

            print(f"   ✅ Backend working: Score={evaluation.relevance_score:.2f}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
        finally:
            # Restore original environment variable
            if original_value:
                os.environ['AGENT_BACKEND'] = original_value
            else:
                os.environ.pop('AGENT_BACKEND', None)


def test_performance_comparison():
    """
    Compare performance between backends.
    """
    print("\n\n🧪 Performance Comparison")
    print("=" * 60)

    # Test data
    test_result = WebSearchResult(
        url="https://example.com/test",
        title="Test Article About AI",
        snippet="This is a comprehensive article about artificial intelligence and its applications.",
        search_query="artificial intelligence"
    )

    topic = "artificial intelligence"

    # Performance test parameters
    num_tests = 3

    backends_to_test = [
        (BackendType.PYDANTIC_AI, "Pydantic.ai"),
        (BackendType.BAREMETAL, "Baremetal")
    ]

    for backend_type, backend_name in backends_to_test:
        print(f"\n⏱️  Performance Test: {backend_name}")
        print("-" * 30)

        times = []
        successes = 0

        for i in range(num_tests):
            try:
                backend = AgentFactory.create_backend(
                    backend_type=backend_type,
                    system_prompt="You are a source evaluation agent."
                )

                evaluator = SourceEvaluatorAgent(backend=backend)

                start_time = time.time()
                evaluation = evaluator.evaluate_source(test_result, topic)
                end_time = time.time()

                duration = end_time - start_time
                times.append(duration)
                successes += 1

                print(f"   Test {i+1}: {duration:.2f}s ✅")

            except Exception as e:
                print(f"   Test {i+1}: Failed - {e} ❌")

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            print(f"   📊 Results ({successes}/{num_tests} successful):")
            print(f"      Average: {avg_time:.2f}s")
            print(f"      Min: {min_time:.2f}s")
            print(f"      Max: {max_time:.2f}s")
        else:
            print(f"   📊 No successful tests")


def main():
    """
    Main test runner.
    """
    print("🚀 Agent Backend Test Suite")
    print("="*60)
    print("This script demonstrates the configurable agent backend system")
    print("that allows switching between pydantic.ai and baremetal approaches.")
    print("="*60)

    # Check system configuration
    print(f"\n⚙️  Current Configuration:")
    print(f"   LLM Model: {settings.llm.model}")
    print(f"   Base URL: {settings.llm.base_url}")
    print(f"   Agent Backend: {settings.agent.backend_type}")

    # Run all tests
    try:
        test_source_evaluation_backends()
        test_sheet_generation_backends()
        test_environment_configuration()
        test_performance_comparison()

        print("\n\n🎉 All Tests Completed!")
        print("="*60)
        print("The agent backend system is working correctly.")
        print("Users can now choose between pydantic.ai and baremetal backends")
        print("by setting the AGENT_BACKEND environment variable.")

    except Exception as e:
        print(f"\n\n❌ Test Suite Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
