"""
Semantic Orchestrator for Enhanced Learning Sheet Generation.

This module combines topic summaries from parallel processing into a
comprehensive, coherent learning sheet using deepseek-r1-32b's 128k context window.
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import EnhancedLearningSheet
from config.settings import settings
from agents.factory import AgentFactory, BackendType
from agents.base import AgentBackend
from agents.summarization_pool import TopicSummary
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger(__name__)


class OrchestrationResponse(BaseModel):
    """Pydantic model for structured LLM response from orchestration."""
    title: str = Field(description="Comprehensive title for the learning sheet")
    content: str = Field(description="Complete 3,000-4,000 word learning sheet content")
    key_takeaways: List[str] = Field(description="5-7 main takeaways from all topics")
    cross_connections: List[str] = Field(description="Relationships identified between topics")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence in orchestration quality")


class SemanticOrchestrator:
    """
    Orchestrates topic summaries into comprehensive learning sheets.

    Combines topic summaries into coherent, educational content with logical flow
    and cross-topic connections. Uses configurable LLM models via baremetal backend.
    """

    def __init__(self, backend: AgentBackend = None, model_name: str = None):
        """
        Initialize the semantic orchestrator.

        Args:
            backend: Backend for LLM calls (uses default model if not specified)
            model_name: LLM model to use for orchestration (uses settings.llm.model if not specified)
        """
        # Use default model from settings if none specified
        if model_name is None:
            model_name = settings.llm.model

        if backend is None:
            system_prompt = (
                "You are an expert educational content orchestrator and technical writer. "
                "Your task is to combine multiple topic summaries into a comprehensive, "
                "coherent learning sheet that flows logically and maintains educational value. "
                "You excel at identifying connections between topics and organizing complex "
                "information into accessible, structured content."
            )
            backend = AgentFactory.create_backend(
                BackendType.BAREMETAL,
                system_prompt,
                model_name=model_name
            )

        self.backend = backend
        self.model_name = model_name

        # Configuration from settings
        self.target_word_count = getattr(settings, 'FINAL_SHEET_TARGET_LENGTH', 4000)
        self.orchestration_timeout = getattr(settings, 'ORCHESTRATION_TIMEOUT', 300)
        self.min_confidence_threshold = getattr(settings, 'CONFIDENCE_SCORE_THRESHOLD', 0.75)

    async def orchestrate_learning_sheet(self,
                                       topic_summaries: List[TopicSummary],
                                       original_topic: str) -> EnhancedLearningSheet:
        """
        Orchestrate topic summaries into a comprehensive learning sheet.

        Args:
            topic_summaries: List of topic summaries from parallel processing
            original_topic: Original topic requested by user

        Returns:
            Enhanced learning sheet with comprehensive content
        """
        start_time = time.time()

        logger.info(f"Starting orchestration for '{original_topic}' with {len(topic_summaries)} topic summaries")

        try:
            # Create comprehensive prompt with all topic summaries
            orchestration_prompt = self._create_orchestration_prompt(topic_summaries, original_topic)

            # Generate the orchestrated learning sheet using LLM
            orchestration_response = await self._call_llm_for_orchestration(orchestration_prompt)

            # Calculate processing statistics
            processing_time = time.time() - start_time
            word_count = self._count_words(orchestration_response.content)

            # Aggregate source information
            all_sources = []
            total_confidence = 0.0
            for summary in topic_summaries:
                all_sources.extend(summary.sources_used)
                total_confidence += summary.confidence_score

            # Remove duplicate sources while preserving order
            unique_sources = list(dict.fromkeys(all_sources))
            avg_topic_confidence = total_confidence / len(topic_summaries) if topic_summaries else 0.0

            # Create topic sections metadata
            topic_sections = [
                {
                    "topic_name": summary.topic.name,
                    "word_count": summary.word_count,
                    "confidence": summary.confidence_score,
                    "key_points_count": len(summary.key_points),
                    "sources_count": len(summary.sources_used)
                }
                for summary in topic_summaries
            ]

            # Calculate semantic processing stats
            semantic_stats = {
                "orchestration_model": self.model_name,
                "context_window_used": len(orchestration_prompt),
                "input_summaries": len(topic_summaries),
                "total_input_words": sum(summary.word_count for summary in topic_summaries),
                "compression_ratio": word_count / sum(summary.word_count for summary in topic_summaries) if topic_summaries else 0,
                "avg_topic_confidence": avg_topic_confidence,
                "orchestration_confidence": orchestration_response.confidence_score,
                "cross_connections_found": len(orchestration_response.cross_connections)
            }

            # Create the enhanced learning sheet
            enhanced_sheet = EnhancedLearningSheet(
                title=orchestration_response.title,
                content=orchestration_response.content,
                key_takeaways=orchestration_response.key_takeaways,
                sources_used=unique_sources,
                confidence_score=min(avg_topic_confidence, orchestration_response.confidence_score),
                topic=original_topic,
                generated_at=datetime.now(),
                total_sources_scraped=len(unique_sources),
                total_content_length=sum(summary.word_count for summary in topic_summaries),

                # Semantic processing fields
                topic_sections=topic_sections,
                semantic_processing_stats=semantic_stats,
                cross_topic_connections=orchestration_response.cross_connections,
                word_count=word_count,
                topics_analyzed=len(topic_summaries),
                topics_summarized=len([s for s in topic_summaries if s.confidence_score >= self.min_confidence_threshold]),
                parallel_processing_time=processing_time
            )

            logger.info(f"Orchestration completed: {word_count} words, {len(unique_sources)} sources, {processing_time:.1f}s")
            return enhanced_sheet

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            # Return a fallback sheet with available information
            return self._create_fallback_sheet(topic_summaries, original_topic, time.time() - start_time)

    def _create_orchestration_prompt(self, topic_summaries: List[TopicSummary], original_topic: str) -> str:
        """Create comprehensive prompt for orchestration with all topic summaries."""

        # Prepare topic summaries for prompt
        topic_content = []
        for i, summary in enumerate(topic_summaries, 1):
            topic_section = f"""
=== TOPIC {i}: {summary.topic.name} ===
Priority: {summary.topic.priority:.2f} | Confidence: {summary.confidence_score:.2f} | Words: {summary.word_count}

SUMMARY:
{summary.summary_content}

KEY POINTS:
{chr(10).join(f"- {point}" for point in summary.key_points)}

SOURCES: {', '.join(summary.sources_used)}
"""
            topic_content.append(topic_section)

        topics_text = "\n".join(topic_content)

        prompt = f"""You are creating a comprehensive learning sheet about "{original_topic}" by orchestrating {len(topic_summaries)} topic summaries into a coherent, educational document.

INPUT TOPIC SUMMARIES:
{topics_text}

Your task is to create a 3,000-4,000 word comprehensive learning sheet that:

1. **LOGICAL ORGANIZATION**: Arrange topics in a logical learning progression
2. **COHERENT FLOW**: Create smooth transitions between topics with connecting ideas
3. **CROSS-TOPIC CONNECTIONS**: Identify and explain relationships between different topics
4. **COMPREHENSIVE COVERAGE**: Ensure all important information from summaries is included
5. **EDUCATIONAL VALUE**: Structure content for optimal learning and understanding
6. **PRACTICAL FOCUS**: Include real-world applications and examples from the sources

RESPONSE FORMAT:
Use this structured text format:

TITLE:
[Create a comprehensive, descriptive title for the learning sheet]

CONTENT:
[Write the complete 3,000-4,000 word learning sheet in Markdown format]
[Organize with clear headings, subheadings, and logical flow]
[Include examples, explanations, and insights from all topic summaries]
[Maintain proper citations and source references]
[Ensure educational progression from basic to advanced concepts]

KEY_TAKEAWAYS:
- [First major takeaway that spans multiple topics]
- [Second major takeaway with practical implications]
- [Third takeaway highlighting key insights]
- [Fourth takeaway about real-world applications]
- [Fifth takeaway about future considerations]

CROSS_CONNECTIONS:
- [Relationship between Topic A and Topic B with explanation]
- [How Topic C builds upon Topic A concepts]
- [Practical connection between Topic D and Topic E]
- [Theoretical foundation linking multiple topics]

CONFIDENCE: [0.0-1.0 score for overall orchestration quality]

REQUIREMENTS:
- Combine ALL topic summaries into cohesive content
- Maintain educational flow and logical progression
- Include specific examples and insights from sources
- Create meaningful connections between topics
- Ensure content is comprehensive yet accessible
- Target 3,000-4,000 words for thorough coverage

Focus on creating university-level educational content that demonstrates deep understanding of "{original_topic}" through semantic orchestration of all provided topics."""

        return prompt

    async def _call_llm_for_orchestration(self, prompt: str) -> OrchestrationResponse:
        """Make LLM call for semantic orchestration - direct synchronous call."""
        try:
            print(f"   🎯 DEBUG: Orchestration prompt ({len(prompt)} chars)")
            print(f"   📝 Prompt preview: {prompt[:200]}...")
            print(f"   🔧 Using model: {self.model_name}")

            # Make direct synchronous call for baremetal approach (no thread pool)
            response = self.backend.generate_response(prompt, OrchestrationResponse)

            print(f"   ✅ DEBUG: Orchestration response received")
            print(f"   📊 Title: {response.title[:100]}...")
            print(f"   📝 Content: {len(response.content)} chars")
            print(f"   🎯 Confidence: {response.confidence_score:.2f}")

            return response

        except Exception as e:
            print(f"   ❌ DEBUG: Orchestration LLM call failed: {str(e)}")
            logger.error(f"Orchestration LLM call failed: {str(e)}")
            raise

    def _count_words(self, text: str) -> int:
        """Count words in text content."""
        return len(text.split())

    def _create_fallback_sheet(self, topic_summaries: List[TopicSummary], original_topic: str, processing_time: float) -> EnhancedLearningSheet:
        """Create fallback learning sheet when orchestration fails."""

        # Combine summaries manually
        fallback_content = f"# {original_topic}\n\n"
        fallback_content += "This learning sheet was generated using a fallback method due to orchestration processing issues.\n\n"

        for i, summary in enumerate(topic_summaries, 1):
            fallback_content += f"## {i}. {summary.topic.name}\n\n"
            fallback_content += f"{summary.summary_content}\n\n"

            if summary.key_points:
                fallback_content += "### Key Points:\n"
                for point in summary.key_points:
                    fallback_content += f"- {point}\n"
                fallback_content += "\n"

        # Aggregate sources
        all_sources = []
        for summary in topic_summaries:
            all_sources.extend(summary.sources_used)
        unique_sources = list(dict.fromkeys(all_sources))

        return EnhancedLearningSheet(
            title=f"{original_topic} - Learning Sheet",
            content=fallback_content,
            key_takeaways=["Fallback generation used", "Topics covered individually", "See sections above for details"],
            sources_used=unique_sources,
            confidence_score=0.3,  # Low confidence for fallback
            topic=original_topic,
            generated_at=datetime.now(),
            total_sources_scraped=len(unique_sources),
            total_content_length=len(fallback_content),
            word_count=self._count_words(fallback_content),
            topics_analyzed=len(topic_summaries),
            topics_summarized=len(topic_summaries),
            parallel_processing_time=processing_time
        )


# Factory function for easy instantiation
def create_semantic_orchestrator(model_name: str = None) -> SemanticOrchestrator:
    """Create a SemanticOrchestrator instance with specified model."""
    # Use default model from settings if none specified
    if model_name is None:
        model_name = settings.llm.model

    system_prompt = (
        "You are an expert educational content orchestrator and technical writer. "
        "Your task is to combine multiple topic summaries into a comprehensive, "
        "coherent learning sheet that flows logically and maintains educational value. "
        "You excel at identifying connections between topics and organizing complex "
        "information into accessible, structured content."
    )
    backend = AgentFactory.create_backend(
        BackendType.BAREMETAL,
        system_prompt,
        model_name=model_name
    )
    return SemanticOrchestrator(backend=backend, model_name=model_name)
