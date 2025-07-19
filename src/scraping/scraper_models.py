"""Scraper-related data models."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class ScrapingRequest(BaseModel):
    """Request to scrape a specific URL."""
    url: str = Field(description="URL to scrape")
    priority: int = Field(default=1, description="Scraping priority (1-5)")
    expected_content_type: str = Field(default="article", description="Expected content type")
    source_evaluation_score: Optional[float] = Field(default=None, description="Quality score from source evaluation")


class ScrapingSession(BaseModel):
    """A complete scraping session."""
    topic: str = Field(description="Topic being researched")
    requests: List[ScrapingRequest] = Field(description="List of scraping requests")
    completed_scrapes: int = Field(default=0, description="Number of completed scrapes")
    successful_scrapes: int = Field(default=0, description="Number of successful scrapes")
    failed_scrapes: int = Field(default=0, description="Number of failed scrapes")
    total_content_length: int = Field(default=0, description="Total content length scraped")
    session_start: datetime = Field(default_factory=datetime.now, description="When scraping started")
    session_end: Optional[datetime] = Field(default=None, description="When scraping completed")


class ScrapingStats(BaseModel):
    """Statistics about scraping performance."""
    total_requests: int = Field(description="Total scraping requests")
    successful_requests: int = Field(description="Successful scraping requests")
    failed_requests: int = Field(description="Failed scraping requests")
    success_rate: float = Field(description="Success rate percentage")
    total_content_length: int = Field(description="Total content length scraped")
    average_content_length: float = Field(description="Average content length per successful scrape")
    total_time_seconds: float = Field(description="Total time spent scraping")
    average_time_per_request: float = Field(description="Average time per scraping request")
