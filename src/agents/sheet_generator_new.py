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

    async def generate_from_content(self, topic: str, scraped_contents: List[ScrapedContent]) -> EnhancedLearningSheet:
        """
        Generate enhanced learning sheet using full scraped content via semantic pipeline.

        This implements the semantic intelligence approach:
        1. Topic Analysis - identify semantic topics from content
        2. Parallel Summarization - create topic-focused summaries
        3. Semantic Orchestration - combine into coherent learning sheet

        Args:
            topic: The topic to create a learning sheet for
            scraped_contents: List of successfully scraped content

        Returns:
            EnhancedLearningSheet with comprehensive content and metadata
        """
        from agents.topic_analyzer import create_topic_analyzer
        from agents.summarization_pool import create_summarization_pool
        from agents.semantic_orchestrator import create_semantic_orchestrator
        from scraping.content_processor import process_scraped_content
        import asyncio
        import time

        start_time = time.time()

        # Step 1: Convert ScrapedContent to ContentChunk format for processing
        content_chunks = process_scraped_content(scraped_contents)

        if not content_chunks:
            # Fallback to snippet-based generation
            print("⚠️  No content chunks available, falling back to snippet-based generation")
            snippet_content = "\n".join([f"Source: {content.title}\n{content.content[:500]}..."
                                       for content in scraped_contents if content.success])
            regular_sheet = self.generate_from_snippets(topic, snippet_content)

            # Convert to EnhancedLearningSheet format
            return EnhancedLearningSheet(
                title=regular_sheet.title,
                content=regular_sheet.content,
                topic=topic,
                sources_used=[content.url for content in scraped_contents if content.success],
                confidence_score=0.5,  # Lower confidence for fallback
                word_count=len(regular_sheet.content.split()),
                topics_analyzed=0,
                topics_summarized=0
            )

        try:
            # Step 2: Topic Analysis - identify semantic topics from content
            print(f"🧠 Step 1: Analyzing semantic topics from {len(content_chunks)} content chunks...")
            analyzer = create_topic_analyzer()
            topic_map = await analyzer.analyze_semantic_topics(content_chunks, topic)
            print(f"   ✅ Identified {len(topic_map.topics)} semantic topics")

            # Step 3: Parallel Summarization - create topic-focused summaries
            print(f"📝 Step 2: Creating parallel topic summaries...")
            max_concurrent = settings.semantic.max_concurrent_summarizers
            pool = create_summarization_pool(max_concurrent=max_concurrent)
            topic_summaries = await pool.create_topic_summaries(topic_map, topic)
            print(f"   ✅ Created {len(topic_summaries)} topic summaries")

            # Step 4: Semantic Orchestration - combine into coherent learning sheet
            print(f"🎯 Step 3: Orchestrating comprehensive learning sheet...")
            orchestrator_model = settings.semantic.orchestration_model
            orchestrator = create_semantic_orchestrator(model_name=orchestrator_model)
            enhanced_sheet = await orchestrator.orchestrate_learning_sheet(topic_summaries, topic)

            processing_time = time.time() - start_time
            enhanced_sheet.parallel_processing_time = processing_time

            print(f"   ✅ Generated {enhanced_sheet.word_count}-word learning sheet in {processing_time:.1f}s")
            print(f"   🎯 Confidence: {enhanced_sheet.confidence_score:.2f}")
            print(f"   📚 Sources: {len(enhanced_sheet.sources_used)}")

            return enhanced_sheet

        except Exception as e:
            print(f"❌ Semantic generation failed: {e}")
            if settings.logging.verbose_output:
                import traceback
                traceback.print_exc()

            # Fallback to snippet-based generation
            print("🔄 Falling back to snippet-based generation...")
            snippet_content = "\n".join([f"Source: {content.title}\n{content.content[:500]}..."
                                       for content in scraped_contents if content.success])
            regular_sheet = self.generate_from_snippets(topic, snippet_content)

            # Convert to EnhancedLearningSheet format with error indication
            return EnhancedLearningSheet(
                title=f"{regular_sheet.title} (Fallback Mode)",
                content=regular_sheet.content + "\n\n*Note: Generated using fallback method due to semantic processing issues.*",
                topic=topic,
                sources_used=[content.url for content in scraped_contents if content.success],
                confidence_score=0.3,  # Low confidence for fallback
                word_count=len(regular_sheet.content.split()),
                topics_analyzed=0,
                topics_summarized=0,
                parallel_processing_time=time.time() - start_time
            )

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


async def generate_learning_sheet_from_content(topic: str, scraped_contents: List[ScrapedContent]) -> EnhancedLearningSheet:
    """
    Generate enhanced learning sheet using full scraped content (semantic approach).

    This is the main entry point for Phase 4 semantic generation that maintains
    compatibility with the existing codebase.
    """
    if settings.logging.verbose_output:
        print(f"📝 Generating enhanced learning sheet using semantic pipeline")
        print(f"   Processing {len(scraped_contents)} scraped sources")
        successful_sources = [c for c in scraped_contents if c.success]
        print(f"   {len(successful_sources)} sources successfully scraped")
        total_chars = sum(len(c.content) for c in successful_sources)
        print(f"   Total content: {total_chars:,} characters")

    # Create agent instance
    generator = SheetGeneratorAgent()

    return await generator.generate_from_content(topic, scraped_contents)
