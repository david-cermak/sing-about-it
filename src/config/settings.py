"""Configuration settings management using environment variables."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            print(f"✅ Loaded environment variables from {env_path}")
        except ImportError:
            print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    else:
        print(f"⚠️  No .env file found at {env_path}")


class LLMConfig(BaseModel):
    """LLM configuration settings."""
    model: str = Field(default="llama3.2", description="LLM model name")
    base_url: str = Field(default="http://127.0.0.1:11434/v1", description="LLM API base URL")
    timeout: int = Field(default=300, description="LLM request timeout in seconds")


class SearchConfig(BaseModel):
    """Search configuration settings."""
    queries_per_topic: int = Field(default=5, description="Number of search queries per topic")
    results_per_query: int = Field(default=3, description="Number of results per search query")
    max_total_results: int = Field(default=15, description="Maximum total search results")


class SourceSelectionConfig(BaseModel):
    """Source selection configuration."""
    max_sources_to_scrape: int = Field(default=8, description="Maximum number of sources to scrape")
    min_relevance_score: float = Field(default=0.6, description="Minimum relevance score for scraping")
    prioritize_recent_content: bool = Field(default=True, description="Prioritize recent content")


class ScrapingConfig(BaseModel):
    """Web scraping configuration."""
    timeout: int = Field(default=30, description="Scraping timeout in seconds")
    delay: int = Field(default=2, description="Delay between requests in seconds")
    max_content_length: int = Field(default=50000, description="Maximum content length in characters")
    respect_robots_txt: bool = Field(default=True, description="Respect robots.txt files")
    user_agent: str = Field(default="LearningSheetGenerator/1.0", description="User agent string")


class ContentProcessingConfig(BaseModel):
    """Content processing configuration."""
    enable_chunking: bool = Field(default=True, description="Enable content chunking")
    max_chunk_size: int = Field(default=10000, description="Maximum chunk size in characters")
    overlap_size: int = Field(default=500, description="Overlap size between chunks")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    log_level: str = Field(default="INFO", description="Logging level")
    verbose_output: bool = Field(default=True, description="Enable verbose output")
    save_scraped_content: bool = Field(default=False, description="Save scraped content to files")


class AgentConfig(BaseModel):
    """Agent backend configuration."""
    backend_type: str = Field(default="pydantic_ai", description="Agent backend type: 'pydantic_ai' or 'baremetal'")


class SemanticConfig(BaseModel):
    """Semantic generation configuration settings."""
    enabled: bool = Field(default=True, description="Enable semantic generation pipeline")
    topic_analysis_model: str = Field(default="llama3.2", description="Model for topic analysis")
    summarization_model: str = Field(default="llama3.2", description="Model for parallel summarization")
    orchestration_model: str = Field(default="deepseek-r1:32b", description="Model for semantic orchestration")
    target_topic_count: int = Field(default=6, description="Target number of semantic topics")
    min_topic_count: int = Field(default=4, description="Minimum number of topics")
    max_topic_count: int = Field(default=8, description="Maximum number of topics")
    max_concurrent_summarizers: int = Field(default=6, description="Maximum concurrent summarizers")
    topic_summary_target_length: int = Field(default=2500, description="Target length for topic summaries")
    final_sheet_target_length: int = Field(default=4000, description="Target length for final learning sheet")
    orchestration_timeout: int = Field(default=300, description="Timeout for orchestration in seconds")
    confidence_threshold: float = Field(default=0.75, description="Minimum confidence threshold")
    fallback_to_snippets: bool = Field(default=True, description="Fallback to snippet-based generation on failure")


class Settings(BaseModel):
    """Main application settings."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    source_selection: SourceSelectionConfig = Field(default_factory=SourceSelectionConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    content_processing: ContentProcessingConfig = Field(default_factory=ContentProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)

    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        load_env()

        return cls(
            llm=LLMConfig(
                model=os.getenv('LLM_MODEL', 'llama3.2'),
                base_url=os.getenv('LLM_BASE_URL', 'http://127.0.0.1:11434/v1'),
                timeout=int(os.getenv('LLM_TIMEOUT', '300'))
            ),
            search=SearchConfig(
                queries_per_topic=int(os.getenv('SEARCH_QUERIES_PER_TOPIC', '5')),
                results_per_query=int(os.getenv('SEARCH_RESULTS_PER_QUERY', '3')),
                max_total_results=int(os.getenv('MAX_TOTAL_SEARCH_RESULTS', '15'))
            ),
            source_selection=SourceSelectionConfig(
                max_sources_to_scrape=int(os.getenv('MAX_SOURCES_TO_SCRAPE', '8')),
                min_relevance_score=float(os.getenv('MIN_SOURCE_RELEVANCE_SCORE', '0.6')),
                prioritize_recent_content=os.getenv('PRIORITIZE_RECENT_CONTENT', 'true').lower() == 'true'
            ),
            scraping=ScrapingConfig(
                timeout=int(os.getenv('SCRAPING_TIMEOUT', '30')),
                delay=int(os.getenv('SCRAPING_DELAY', '2')),
                max_content_length=int(os.getenv('MAX_CONTENT_LENGTH', '50000')),
                respect_robots_txt=os.getenv('RESPECT_ROBOTS_TXT', 'true').lower() == 'true',
                user_agent=os.getenv('USER_AGENT', 'LearningSheetGenerator/1.0')
            ),
            content_processing=ContentProcessingConfig(
                enable_chunking=os.getenv('ENABLE_CONTENT_CHUNKING', 'true').lower() == 'true',
                max_chunk_size=int(os.getenv('MAX_CHUNK_SIZE', '10000')),
                overlap_size=int(os.getenv('OVERLAP_SIZE', '500'))
            ),
            logging=LoggingConfig(
                log_level=os.getenv('LOG_LEVEL', 'INFO'),
                verbose_output=os.getenv('VERBOSE_OUTPUT', 'true').lower() == 'true',
                save_scraped_content=os.getenv('SAVE_SCRAPED_CONTENT', 'false').lower() == 'true'
            ),
            agent=AgentConfig(
                backend_type=os.getenv('AGENT_BACKEND', 'pydantic_ai')
            ),
            semantic=SemanticConfig(
                enabled=os.getenv('SEMANTIC_GENERATION_ENABLED', 'true').lower() == 'true',
                topic_analysis_model=os.getenv('TOPIC_ANALYSIS_MODEL', 'llama3.2'),
                summarization_model=os.getenv('SUMMARIZATION_MODEL', 'llama3.2'),
                orchestration_model=os.getenv('ORCHESTRATION_MODEL', 'deepseek-r1:32b'),
                target_topic_count=int(os.getenv('TARGET_TOPIC_COUNT', '6')),
                min_topic_count=int(os.getenv('MIN_TOPIC_COUNT', '4')),
                max_topic_count=int(os.getenv('MAX_TOPIC_COUNT', '8')),
                max_concurrent_summarizers=int(os.getenv('MAX_CONCURRENT_SUMMARIZERS', '6')),
                topic_summary_target_length=int(os.getenv('TOPIC_SUMMARY_TARGET_LENGTH', '2500')),
                final_sheet_target_length=int(os.getenv('FINAL_SHEET_TARGET_LENGTH', '4000')),
                orchestration_timeout=int(os.getenv('ORCHESTRATION_TIMEOUT', '300')),
                confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.75')),
                fallback_to_snippets=os.getenv('FALLBACK_TO_SNIPPETS', 'true').lower() == 'true'
            )
        )


# Global settings instance
settings = Settings.from_env()
