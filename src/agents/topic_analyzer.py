"""
Topic Analysis Engine for Semantic Learning Sheet Generation.

This module analyzes ContentChunk[] from Phase 3 to identify semantic topics
and map chunks to topics for parallel processing.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import ContentChunk, ScrapedContent
from config.settings import settings
from agents.factory import create_sheet_generator_backend, AgentFactory, BackendType
from agents.base import AgentBackend
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger(__name__)


class TopicIdentificationResponse(BaseModel):
    """Pydantic model for LLM response when identifying semantic topics."""
    topics: List[Dict[str, Any]] = Field(description="List of identified topics with name, description, complexity, priority")
    reasoning: str = Field(description="Reasoning for topic identification")


class ChunkMappingResponse(BaseModel):
    """Pydantic model for LLM response when mapping chunks to topics."""
    mappings: List[Dict[str, Any]] = Field(description="Chunk to topic mappings with scores")
    analysis: Optional[str] = Field(default="", description="Analysis of the mapping decisions")


class SemanticTopic:
    """Represents a semantic topic identified from content analysis."""

    def __init__(self, name: str, description: str, complexity: str = "intermediate", priority: float = 1.0):
        self.name = name
        self.description = description
        self.complexity = complexity  # beginner, intermediate, advanced
        self.priority = priority  # 0.0 to 1.0
        self.chunk_mappings: List[ChunkMapping] = []

    def add_chunk_mapping(self, chunk: ContentChunk, relevance_score: float, reasoning: str):
        """Add a chunk mapping to this topic."""
        mapping = ChunkMapping(
            chunk=chunk,
            topic=self,
            relevance_score=relevance_score,
            reasoning=reasoning
        )
        self.chunk_mappings.append(mapping)

    def get_total_content_length(self) -> int:
        """Get total character count for all chunks in this topic."""
        return sum(len(mapping.chunk.content) for mapping in self.chunk_mappings)

    def get_chunk_count(self) -> int:
        """Get number of chunks assigned to this topic."""
        return len(self.chunk_mappings)


class ChunkMapping:
    """Maps a ContentChunk to a SemanticTopic with relevance scoring."""

    def __init__(self, chunk: ContentChunk, topic: SemanticTopic, relevance_score: float, reasoning: str):
        self.chunk = chunk
        self.topic = topic
        self.relevance_score = relevance_score  # 0.0 to 1.0
        self.reasoning = reasoning


class TopicMap:
    """Complete mapping of chunks to semantic topics."""

    def __init__(self, topics: List[SemanticTopic], coverage_analysis: str):
        self.topics = topics
        self.coverage_analysis = coverage_analysis
        self.topic_relationships: List[str] = []

    def get_chunks_for_topic(self, topic_name: str) -> List[ContentChunk]:
        """Get all chunks assigned to a specific topic."""
        for topic in self.topics:
            if topic.name == topic_name:
                return [mapping.chunk for mapping in topic.chunk_mappings]
        return []

    def get_orphaned_chunks(self, all_chunks: List[ContentChunk]) -> List[ContentChunk]:
        """Find chunks that weren't assigned to any topic."""
        assigned_chunks = set()
        for topic in self.topics:
            for mapping in topic.chunk_mappings:
                assigned_chunks.add(id(mapping.chunk))

        return [chunk for chunk in all_chunks if id(chunk) not in assigned_chunks]

    def validate_coverage(self, all_chunks: List[ContentChunk]) -> Dict[str, any]:
        """Validate that topic coverage is comprehensive."""
        orphaned = self.get_orphaned_chunks(all_chunks)
        total_assigned = sum(topic.get_chunk_count() for topic in self.topics)

        return {
            "total_chunks": len(all_chunks),
            "assigned_chunks": total_assigned,
            "orphaned_chunks": len(orphaned),
            "coverage_percentage": (total_assigned / len(all_chunks)) * 100 if all_chunks else 0,
            "orphaned_chunk_urls": [chunk.source_url for chunk in orphaned]
        }


class TopicAnalyzer:
    """
    Analyzes ContentChunk[] to identify semantic topics and create mappings.

    Uses LLM to understand content meaning and group related chunks into
    coherent topics for parallel summarization.
    """

    def __init__(self, backend: AgentBackend = None):
        """Initialize the topic analyzer with configurable backend."""
        # Use baremetal backend to avoid async event loop conflicts
        if backend is None:
            system_prompt = (
                "You are an expert content analyst specializing in semantic topic identification. "
                "Your task is to analyze content and identify coherent topics for educational purposes. "
                "Always respond with valid JSON containing the requested structure."
            )
            backend = AgentFactory.create_backend(BackendType.BAREMETAL, system_prompt)

        self.backend = backend
        self.target_topic_count = getattr(settings, 'TARGET_TOPIC_COUNT', 6)
        self.min_topic_count = getattr(settings, 'MIN_TOPIC_COUNT', 4)
        self.max_topic_count = getattr(settings, 'MAX_TOPIC_COUNT', 8)

    async def analyze_semantic_topics(self, content_chunks: List[ContentChunk], original_topic: str) -> TopicMap:
        """
        Analyze content chunks to identify semantic topics.

        Args:
            content_chunks: List of ContentChunk objects from Phase 3
            original_topic: Original search topic for context

        Returns:
            TopicMap with semantic topics and chunk assignments
        """
        logger.info(f"Starting topic analysis for {len(content_chunks)} chunks on topic: {original_topic}")

        try:
            # Step 1: Get content overview for topic identification
            overview = await self._create_content_overview(content_chunks, original_topic)

            # Step 2: Identify semantic topics using LLM
            topics = await self._identify_semantic_topics(overview, original_topic)

            # Step 3: Map chunks to topics
            topic_map = await self._map_chunks_to_topics(content_chunks, topics, original_topic)

            # Step 4: Validate and optimize mappings
            await self._validate_and_optimize_mappings(topic_map, content_chunks)

            logger.info(f"Topic analysis complete: {len(topic_map.topics)} topics identified")
            return topic_map

        except Exception as e:
            logger.error(f"Topic analysis failed: {str(e)}")
            # Return fallback topic map
            return self._create_fallback_topic_map(content_chunks, original_topic)

    async def _create_content_overview(self, chunks: List[ContentChunk], original_topic: str) -> str:
        """Create a content overview from chunk samples for topic identification."""
        # Take samples from chunks to create overview (within token limits)
        sample_size = min(len(chunks), 20)  # Limit to 20 chunks for overview
        sampled_chunks = chunks[:sample_size]

        overview_parts = []
        for i, chunk in enumerate(sampled_chunks):
            # Take first ~500 chars from each chunk for overview
            sample = chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content
            overview_parts.append(f"Chunk {i+1} (from {chunk.source_url}):\n{sample}\n")

        overview = "\n".join(overview_parts)

        # Ensure overview isn't too long
        if len(overview) > 15000:  # ~4k tokens
            overview = overview[:15000] + "\n[Content truncated for analysis...]"

        return overview

    async def _identify_semantic_topics(self, content_overview: str, original_topic: str) -> List[SemanticTopic]:
        """Use LLM to identify semantic topics from content overview."""

        prompt = f"""Analyze the following content overview about "{original_topic}" and identify {self.target_topic_count} distinct semantic topics that would organize this content for educational purposes.

CONTENT OVERVIEW:
{content_overview}

REQUIREMENTS:
- Identify {self.min_topic_count} to {self.max_topic_count} semantic topics
- Topics should be educationally logical and comprehensive
- Each topic should represent a distinct aspect of "{original_topic}"
- Topics should flow in a logical learning progression
- Consider different complexity levels (beginner to advanced concepts)

For each topic, provide:
1. Topic Name (concise, descriptive)
2. Description (2-3 sentences explaining what this topic covers)
3. Complexity Level (beginner, intermediate, advanced)
4. Educational Priority (1-5 scale, how important for understanding the subject)

Format as JSON:
{{
  "topics": [
    {{
      "name": "Topic Name",
      "description": "Detailed description of what this topic covers...",
      "complexity": "intermediate",
      "priority": 4
    }}
  ]
}}

Focus on creating a comprehensive educational framework that covers all major aspects of the content."""

        try:
            print(f"   🔍 DEBUG: Topic identification prompt ({len(prompt)} chars)")
            print(f"   📝 Prompt preview: {prompt[:200]}...")

            # Direct structured response - no JSON conversion needed!
            response = await self._call_llm_with_response_type(prompt, TopicIdentificationResponse)

            print(f"   ✅ DEBUG: Topic identification response received")
            print(f"   📊 Found {len(response.topics)} topics")

            # Use structured response directly
            topics = []
            for topic_info in response.topics:
                topic = SemanticTopic(
                    name=topic_info['name'],
                    description=topic_info['description'],
                    complexity=topic_info.get('complexity', 'intermediate'),
                    priority=float(topic_info.get('priority', 3)) / 5.0  # Normalize to 0-1
                )
                topics.append(topic)

            logger.info(f"Identified {len(topics)} semantic topics: {[t.name for t in topics]}")
            return topics

        except Exception as e:
            logger.error(f"LLM topic identification failed: {str(e)}")
            # Return fallback topics
            return self._create_fallback_topics(original_topic)

    async def _map_chunks_to_topics(self, chunks: List[ContentChunk], topics: List[SemanticTopic], original_topic: str) -> TopicMap:
        """Map content chunks to identified semantic topics."""

        logger.info(f"Mapping {len(chunks)} chunks to {len(topics)} topics")

        # Process chunks in batches to avoid token limits
        batch_size = 10  # Process 10 chunks at a time

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            await self._map_chunk_batch_to_topics(batch_chunks, topics, original_topic)

        # Create coverage analysis
        coverage_analysis = await self._analyze_topic_coverage(topics, chunks)

        return TopicMap(topics=topics, coverage_analysis=coverage_analysis)

    async def _map_chunk_batch_to_topics(self, chunk_batch: List[ContentChunk], topics: List[SemanticTopic], original_topic: str):
        """Map a batch of chunks to topics using LLM analysis."""

        # Create topic descriptions for context
        topic_descriptions = []
        for i, topic in enumerate(topics):
            topic_descriptions.append(f"{i+1}. {topic.name}: {topic.description}")

        topics_text = "\n".join(topic_descriptions)

        # Create chunk summaries for analysis
        chunk_summaries = []
        for i, chunk in enumerate(chunk_batch):
            # Truncate chunk content for analysis
            content_preview = chunk.content[:1000] + "..." if len(chunk.content) > 1000 else chunk.content
            chunk_summaries.append(f"Chunk {i+1} (from {chunk.source_url}):\n{content_preview}")

        chunks_text = "\n\n".join(chunk_summaries)

        prompt = f"""Map the following content chunks to the most relevant semantic topics for "{original_topic}".

SEMANTIC TOPICS:
{topics_text}

CONTENT CHUNKS TO MAP:
{chunks_text}

For each chunk, determine:
1. Most relevant topic (1-{len(topics)})
2. Relevance score (0.0-1.0, how well the chunk fits the topic)
3. Brief reasoning for the mapping

PREFERRED FORMAT (simple text):
CHUNK_MAPPINGS:
chunk 1 → topic 2 (score: 0.85) - discusses implementation details for practical applications
chunk 2 → topic 1 (score: 0.90) - covers fundamental security concepts
chunk 3 → topic 2 (score: 0.75) - shows real-world security measures

ANALYSIS: Brief summary of mapping decisions and overall content distribution.

ALTERNATIVE JSON FORMAT (if text fails):
{{
  "mappings": [
    {{
      "chunk_index": 1,
      "topic_number": 2,
      "relevance_score": 0.85,
      "reasoning": "discusses implementation details for practical applications"
    }}
  ],
  "analysis": "Brief summary of mapping decisions"
}}

Guidelines:
- Each chunk should map to exactly one primary topic
- Relevance scores should reflect how well content fits topic
- Use either format - text preferred for reliability
- Consider educational value and topic coherence"""

        try:
            print(f"   🔍 DEBUG: Chunk mapping prompt ({len(prompt)} chars)")
            print(f"   📝 Prompt preview: {prompt[:200]}...")

            # Direct structured response - no JSON conversion needed!
            response = await self._call_llm_with_response_type(prompt, ChunkMappingResponse)

            print(f"   ✅ DEBUG: Chunk mapping response received")
            print(f"   📊 Found {len(response.mappings)} mappings for {len(chunk_batch)} chunks")
            if len(response.mappings) < len(chunk_batch):
                print(f"   ⚠️  WARNING: Expected {len(chunk_batch)} mappings, got {len(response.mappings)}")

            # Use structured response directly
            for mapping_info in response.mappings:
                chunk_index = mapping_info['chunk_index'] - 1  # Convert to 0-based
                topic_number = mapping_info['topic_number'] - 1  # Convert to 0-based

                if 0 <= chunk_index < len(chunk_batch) and 0 <= topic_number < len(topics):
                    chunk = chunk_batch[chunk_index]
                    topic = topics[topic_number]
                    relevance_score = float(mapping_info['relevance_score'])
                    reasoning = mapping_info['reasoning']

                    topic.add_chunk_mapping(chunk, relevance_score, reasoning)
                    print(f"   ✅ Mapped chunk {chunk_index+1} → topic {topic_number+1} (score: {relevance_score})")
                else:
                    print(f"   ❌ Invalid mapping: chunk {chunk_index+1}, topic {topic_number+1} (batch size: {len(chunk_batch)}, topics: {len(topics)})")

        except Exception as e:
            print(f"   ❌ DEBUG: Chunk mapping failed for batch: {str(e)}")
            logger.error(f"Chunk mapping failed for batch: {str(e)}")
            # Fallback: distribute chunks evenly across topics
            print(f"   🔄 Using fallback: distributing {len(chunk_batch)} chunks across {len(topics)} topics")
            for i, chunk in enumerate(chunk_batch):
                topic_index = i % len(topics)
                topics[topic_index].add_chunk_mapping(chunk, 0.5, "Fallback assignment")
                print(f"   📍 Fallback: chunk {i+1} → topic {topic_index+1}")

    async def _analyze_topic_coverage(self, topics: List[SemanticTopic], all_chunks: List[ContentChunk]) -> str:
        """Analyze how well topics cover the content."""

        total_chunks = len(all_chunks)
        assigned_chunks = sum(topic.get_chunk_count() for topic in topics)

        topic_stats = []
        for topic in topics:
            stats = f"- {topic.name}: {topic.get_chunk_count()} chunks ({topic.get_total_content_length():,} chars)"
            topic_stats.append(stats)

        coverage_analysis = f"""Topic Coverage Analysis:
Total chunks: {total_chunks}
Assigned chunks: {assigned_chunks}
Coverage: {(assigned_chunks/total_chunks)*100:.1f}%

Topic Distribution:
{chr(10).join(topic_stats)}

This analysis shows how content is distributed across semantic topics for parallel processing."""

        return coverage_analysis

    async def _validate_and_optimize_mappings(self, topic_map: TopicMap, all_chunks: List[ContentChunk]):
        """Validate topic mappings and optimize if needed."""

        validation = topic_map.validate_coverage(all_chunks)
        logger.info(f"Topic coverage validation: {validation}")

        # If coverage is poor, log warning
        if validation['coverage_percentage'] < 90:
            logger.warning(f"Low topic coverage: {validation['coverage_percentage']:.1f}%. {validation['orphaned_chunks']} chunks unassigned.")

        # Balance topic sizes if very uneven
        avg_chunks_per_topic = len(all_chunks) / len(topic_map.topics)
        for topic in topic_map.topics:
            chunk_count = topic.get_chunk_count()
            if chunk_count < avg_chunks_per_topic * 0.3:  # Topic has very few chunks
                logger.warning(f"Topic '{topic.name}' has only {chunk_count} chunks (avg: {avg_chunks_per_topic:.1f})")

    def _create_fallback_topic_map(self, chunks: List[ContentChunk], original_topic: str) -> TopicMap:
        """Create a fallback topic map if analysis fails."""

        logger.warning("Creating fallback topic map due to analysis failure")

        fallback_topics = self._create_fallback_topics(original_topic)

        # Distribute chunks evenly across fallback topics
        for i, chunk in enumerate(chunks):
            topic_index = i % len(fallback_topics)
            fallback_topics[topic_index].add_chunk_mapping(chunk, 0.5, "Fallback assignment")

        coverage_analysis = f"Fallback topic assignment: {len(chunks)} chunks distributed across {len(fallback_topics)} generic topics."

        return TopicMap(topics=fallback_topics, coverage_analysis=coverage_analysis)

    def _create_fallback_topics(self, original_topic: str) -> List[SemanticTopic]:
        """Create generic fallback topics."""

        fallback_topics = [
            SemanticTopic("Fundamentals", f"Basic concepts and fundamentals of {original_topic}", "beginner", 1.0),
            SemanticTopic("Core Concepts", f"Core concepts and principles of {original_topic}", "intermediate", 0.9),
            SemanticTopic("Applications", f"Practical applications and use cases of {original_topic}", "intermediate", 0.8),
            SemanticTopic("Advanced Topics", f"Advanced topics and specialized areas of {original_topic}", "advanced", 0.7),
            SemanticTopic("Best Practices", f"Best practices and recommendations for {original_topic}", "intermediate", 0.8),
            SemanticTopic("Case Studies", f"Real-world examples and case studies of {original_topic}", "intermediate", 0.7)
        ]

        return fallback_topics[:self.target_topic_count]  # Return only target number of topics

    async def _call_llm_with_response_type(self, prompt: str, response_type) -> Any:
        """Make LLM call with specific response type - direct synchronous call."""
        try:
            # Make direct synchronous call for baremetal approach (no thread pool)
            response = self.backend.generate_response(prompt, response_type)
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise

    # REMOVED: No longer needed - using structured objects directly

    # REMOVED: No longer needed - using structured objects directly


# Factory function for easy instantiation
def create_topic_analyzer() -> TopicAnalyzer:
    """Create a TopicAnalyzer instance with default configuration using baremetal backend."""
    # Explicitly use baremetal backend to avoid async conflicts
    system_prompt = (
        "You are an expert content analyst specializing in semantic topic identification. "
        "Your task is to analyze content and identify coherent topics for educational purposes. "
        "Always respond with valid JSON containing the requested structure."
    )
    backend = AgentFactory.create_backend(BackendType.BAREMETAL, system_prompt)
    return TopicAnalyzer(backend=backend)
