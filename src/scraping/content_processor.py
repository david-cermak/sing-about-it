"""Content processing and cleaning functionality."""

from typing import List
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import ScrapedContent, ContentChunk
from config.settings import settings


def clean_content(content: str) -> str:
    """
    Clean and process scraped content.

    TODO: Implement in Phase 3
    - Remove HTML tags and navigation
    - Extract main article content
    - Convert to markdown
    - Handle encoding issues
    """
    # Placeholder implementation
    return content


def chunk_content(scraped_content: ScrapedContent) -> List[ContentChunk]:
    """
    Split content into manageable chunks for LLM processing.

    TODO: Implement in Phase 3
    - Intelligent chunking based on content structure
    - Overlap between chunks for context
    - Preserve source attribution
    """
    # Placeholder implementation
    return [
        ContentChunk(
            source_url=scraped_content.url,
            chunk_index=0,
            content=scraped_content.content,
            word_count=len(scraped_content.content.split()),
            overlap_with_next=False
        )
    ]


def process_scraped_content(scraped_contents: List[ScrapedContent]) -> List[ContentChunk]:
    """
    Process multiple scraped contents into chunks.

    TODO: Implement in Phase 3
    """
    all_chunks = []
    for content in scraped_contents:
        if content.success:
            cleaned_content = clean_content(content.content)
            content.content = cleaned_content
            chunks = chunk_content(content)
            all_chunks.extend(chunks)

    return all_chunks
