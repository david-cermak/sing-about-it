"""Base Pydantic models for data structures used across the application."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime


class WebSearchResult(BaseModel):
    """Result from a web search query."""
    url: str = Field(description="URL of the search result")
    title: str = Field(description="Title of the search result")
    snippet: str = Field(description="Short snippet/summary of the content")
    search_query: Optional[str] = Field(default=None, description="Original search query that found this result")


class SourceEvaluation(BaseModel):
    """Evaluation of a web source for relevance and quality."""
    url: str = Field(description="URL of the source")
    title: str = Field(description="Title of the source")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance to topic (0-1)")
    authority_score: float = Field(ge=0.0, le=1.0, description="Domain authority (0-1)")
    content_type: str = Field(description="Type: academic, news, blog, commercial, documentation")
    should_scrape: bool = Field(description="Whether to scrape this source")
    reasoning: str = Field(description="Why this source was selected/rejected")
    estimated_quality: str = Field(description="Estimated quality: high, medium, low")


class ScrapedContent(BaseModel):
    """Content scraped from a web source."""
    url: str = Field(description="URL of the scraped source")
    title: str = Field(description="Title of the content")
    content: str = Field(description="Main text content")
    content_length: int = Field(description="Length of content in characters")
    scraped_at: datetime = Field(default_factory=datetime.now, description="When the content was scraped")
    success: bool = Field(description="Whether scraping was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if scraping failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the content")


class ContentChunk(BaseModel):
    """A chunk of content for processing by the LLM."""
    source_url: str = Field(description="URL of the original source")
    chunk_index: int = Field(description="Index of this chunk in the source")
    content: str = Field(description="Text content of the chunk")
    word_count: int = Field(description="Number of words in the chunk")
    overlap_with_next: bool = Field(default=False, description="Whether this chunk overlaps with the next")


class LearningSheet(BaseModel):
    """Basic learning sheet model (current version)."""
    title: str = Field(description="Title of the learning sheet")
    content: str = Field(description="~1000 word report in Markdown")


class EnhancedLearningSheet(BaseModel):
    """Enhanced learning sheet model with semantic processing metadata (Phase 4)."""
    title: str = Field(description="Title of the learning sheet")
    content: str = Field(description="3,000-4,000 word comprehensive report in Markdown")
    key_takeaways: List[str] = Field(description="5-7 bullet point takeaways", default_factory=list)
    sources_used: List[str] = Field(description="URLs of sources actually referenced", default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in information accuracy", default=0.0)
    topic: str = Field(description="Original topic requested")
    generated_at: datetime = Field(default_factory=datetime.now, description="When the sheet was generated")
    total_sources_scraped: int = Field(description="Number of sources successfully scraped", default=0)
    total_content_length: int = Field(description="Total length of source content used", default=0)

    # New semantic processing fields
    topic_sections: List[Dict[str, Any]] = Field(
        description="Organized semantic topic sections with metadata",
        default_factory=list
    )
    semantic_processing_stats: Dict[str, Any] = Field(
        description="Statistics from semantic analysis and orchestration",
        default_factory=dict
    )
    cross_topic_connections: List[str] = Field(
        description="Identified relationships and connections between topics",
        default_factory=list
    )
    word_count: int = Field(description="Actual word count of generated content", default=0)
    topics_analyzed: int = Field(description="Number of semantic topics identified", default=0)
    topics_summarized: int = Field(description="Number of topics successfully summarized", default=0)
    parallel_processing_time: float = Field(description="Time saved through parallel processing", default=0.0)


class ProcessingStats(BaseModel):
    """Statistics about the processing pipeline."""
    total_search_results: int = Field(description="Total number of search results found")
    sources_evaluated: int = Field(description="Number of sources evaluated")
    sources_selected_for_scraping: int = Field(description="Number of sources selected for scraping")
    sources_successfully_scraped: int = Field(description="Number of sources successfully scraped")
    total_content_length: int = Field(description="Total length of scraped content")
    processing_time_seconds: float = Field(description="Total processing time in seconds")
    search_time_seconds: float = Field(description="Time spent searching")
    scraping_time_seconds: float = Field(description="Time spent scraping")
    llm_generation_time_seconds: float = Field(description="Time spent generating the learning sheet")
