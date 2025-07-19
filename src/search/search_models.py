"""Search-related data models."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import WebSearchResult


class SearchQuery(BaseModel):
    """A search query with metadata."""
    query: str = Field(description="The search query string")
    topic: str = Field(description="The main topic this query relates to")
    query_type: str = Field(description="Type of query: overview, recent, best_practices, examples, trends")
    priority: int = Field(default=1, description="Priority of this query (1-5)")


class SearchSession(BaseModel):
    """A complete search session for a topic."""
    topic: str = Field(description="Main topic being searched")
    queries: List[SearchQuery] = Field(description="List of search queries")
    results: Dict[str, List[WebSearchResult]] = Field(description="Results indexed by query string")
    total_results: int = Field(description="Total number of results found")
    search_duration_seconds: float = Field(description="Time taken to complete all searches")
    success_rate: float = Field(description="Percentage of successful searches")


class SearchStats(BaseModel):
    """Statistics about search performance."""
    total_queries: int = Field(description="Total number of queries executed")
    successful_queries: int = Field(description="Number of successful queries")
    failed_queries: int = Field(description="Number of failed queries")
    total_results: int = Field(description="Total results found across all queries")
    average_results_per_query: float = Field(description="Average number of results per query")
    total_time_seconds: float = Field(description="Total time spent searching")
