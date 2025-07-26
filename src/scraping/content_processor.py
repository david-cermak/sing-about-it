"""Content processing and cleaning functionality."""

import re
from typing import List
import sys
from pathlib import Path
import logging
from markdownify import markdownify as md

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import ScrapedContent, ContentChunk
from config.settings import settings

# Setup logging
logger = logging.getLogger(__name__)


def clean_content(content: str) -> str:
    """
    Clean and process scraped content.

    Removes unwanted characters, normalizes whitespace, and improves readability.
    """
    if not content:
        return ""

    # Remove common unwanted patterns
    patterns_to_remove = [
        r'\s*\[?\s*Advertisement\s*\]?\s*',  # Advertisement text
        r'\s*\[?\s*Sponsored\s*\]?\s*',     # Sponsored content
        r'\s*Click here to\s+.*?\.?\s*',    # Click here links
        r'\s*Subscribe\s+to\s+.*?\.?\s*',   # Subscribe prompts
        r'\s*Follow us on\s+.*?\.?\s*',     # Social media follows
        r'\s*Share this\s+.*?\.?\s*',       # Share prompts
        r'\s*Read more\s*:?\s*.*?\.?\s*',   # Read more links
        r'\s*Continue reading\s*.*?\.?\s*', # Continue reading
        r'\s*\[.*?\]\s*',                   # Content in square brackets (often ads/nav)
    ]

    for pattern in patterns_to_remove:
        content = re.sub(pattern, ' ', content, flags=re.IGNORECASE)

    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content)  # Multiple spaces to single space
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines to double

    # Clean up punctuation
    content = re.sub(r'\.{3,}', '...', content)  # Multiple dots to ellipsis
    content = re.sub(r'[^\w\s.,!?;:()\-\'""\u2019\u2018\u201c\u201d]', ' ', content)  # Keep only basic punctuation

    # Remove excessive punctuation
    content = re.sub(r'[!?]{2,}', '!', content)  # Multiple exclamation/question marks

    # Fix common encoding issues
    content = content.replace('\u2019', "'")  # Right single quotation mark
    content = content.replace('\u2018', "'")  # Left single quotation mark
    content = content.replace('\u201c', '"')  # Left double quotation mark
    content = content.replace('\u201d', '"')  # Right double quotation mark
    content = content.replace('\u2013', '-')  # En dash
    content = content.replace('\u2014', '-')  # Em dash
    content = content.replace('\u00a0', ' ')  # Non-breaking space

    # Final cleanup
    content = content.strip()

    return content


def extract_main_sentences(content: str, min_sentence_length: int = 10) -> List[str]:
    """
    Extract meaningful sentences from content.

    Filters out very short sentences that are likely navigation or metadata.
    """
    if not content:
        return []

    # Split into sentences (basic approach)
    sentences = re.split(r'[.!?]+\s+', content)

    # Filter meaningful sentences
    meaningful_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if (len(sentence) >= min_sentence_length and
            not re.match(r'^(home|menu|navigation|subscribe|follow|click|share)', sentence.lower())):
            meaningful_sentences.append(sentence)

    return meaningful_sentences


def calculate_reading_complexity(content: str) -> float:
    """
    Calculate a simple reading complexity score (0-1, where 1 is most complex).

    Based on sentence length and word complexity.
    """
    if not content:
        return 0.0

    sentences = extract_main_sentences(content)
    if not sentences:
        return 0.0

    # Calculate average sentence length
    words = content.split()
    avg_sentence_length = len(words) / len(sentences) if sentences else 0

    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

    # Normalize to 0-1 scale (rough heuristic)
    sentence_complexity = min(avg_sentence_length / 30, 1.0)  # 30 words = max complexity
    word_complexity = min(avg_word_length / 10, 1.0)  # 10 chars = max complexity

    return (sentence_complexity + word_complexity) / 2


def chunk_content(scraped_content: ScrapedContent,
                 max_chunk_size: int = 8000,
                 overlap_size: int = 500) -> List[ContentChunk]:
    """
    Split content into manageable chunks for LLM processing.

    Uses intelligent chunking based on content structure with overlap for context.
    """
    if not scraped_content.success or not scraped_content.content:
        return []

    content = scraped_content.content
    sentences = extract_main_sentences(content, min_sentence_length=5)

    if not sentences:
        # Fallback to simple word-based chunking
        words = content.split()
        max_words_per_chunk = max_chunk_size // 6  # Rough estimate: 6 chars per word

        chunks = []
        for i in range(0, len(words), max_words_per_chunk):
            chunk_words = words[i:i + max_words_per_chunk]
            chunk_content = ' '.join(chunk_words)

            chunks.append(ContentChunk(
                source_url=scraped_content.url,
                chunk_index=len(chunks),
                content=chunk_content,
                word_count=len(chunk_words),
                overlap_with_next=i + max_words_per_chunk < len(words)
            ))

        return chunks

    # Sentence-based chunking with overlap
    chunks = []
    current_chunk = []
    current_length = 0

    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)

        # If adding this sentence would exceed chunk size, finalize current chunk
        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunk_content = '. '.join(current_chunk) + '.'

            chunks.append(ContentChunk(
                source_url=scraped_content.url,
                chunk_index=len(chunks),
                content=chunk_content,
                word_count=len(chunk_content.split()),
                overlap_with_next=i < len(sentences)
            ))

            # Start new chunk with overlap from previous chunk
            if overlap_size > 0 and current_chunk:
                overlap_text = chunk_content[-overlap_size:]
                # Find the start of a sentence in the overlap
                overlap_sentences = re.split(r'[.!?]+\s+', overlap_text)
                if len(overlap_sentences) > 1:
                    current_chunk = [overlap_sentences[-1]]  # Start with last complete sentence
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    # Handle remaining content
    if current_chunk:
        chunk_content = '. '.join(current_chunk) + '.'
        chunks.append(ContentChunk(
            source_url=scraped_content.url,
            chunk_index=len(chunks),
            content=chunk_content,
            word_count=len(chunk_content.split()),
            overlap_with_next=False
        ))

    logger.info(f"Split content from {scraped_content.url} into {len(chunks)} chunks")
    return chunks


def validate_content_quality(scraped_content: ScrapedContent) -> bool:
    """
    Validate that scraped content meets quality thresholds.

    Checks for minimum length, language, and content structure.
    """
    if not scraped_content.success or not scraped_content.content:
        return False

    content = scraped_content.content.strip()

    # Minimum length check
    if len(content) < 200:  # At least 200 characters
        logger.warning(f"Content too short for {scraped_content.url}: {len(content)} chars")
        return False

    # Check for reasonable sentence structure
    sentences = extract_main_sentences(content, min_sentence_length=5)
    if len(sentences) < 3:  # At least 3 meaningful sentences
        logger.warning(f"Too few sentences for {scraped_content.url}: {len(sentences)}")
        return False

    # Check for repeated content (possible scraping error) - made more lenient
    words = content.split()
    unique_words = set(words)
    uniqueness_ratio = len(unique_words) / len(words) if words else 0
    if uniqueness_ratio < 0.2:  # Reduced from 0.3 to 0.2 (20% unique words minimum)
        logger.warning(f"Content appears highly repetitive for {scraped_content.url}: {uniqueness_ratio:.2f} uniqueness ratio")
        return False
    elif uniqueness_ratio < 0.3:  # Log warning for borderline content
        logger.info(f"Content has low uniqueness for {scraped_content.url}: {uniqueness_ratio:.2f} ratio, but proceeding")

    # Check for reasonable text (not just numbers/symbols)
    text_chars = sum(1 for char in content if char.isalpha())
    text_ratio = text_chars / len(content) if content else 0
    if text_ratio < 0.4:  # Reduced from 0.5 to 0.4 (40% alphabetic characters minimum)
        logger.warning(f"Content doesn't appear to be normal text for {scraped_content.url}: {text_ratio:.2f} text ratio")
        return False

    return True


def process_scraped_content(scraped_contents: List[ScrapedContent]) -> List[ContentChunk]:
    """
    Process multiple scraped contents into cleaned and chunked format.

    Filters out low-quality content and creates optimal chunks for LLM processing.
    """
    all_chunks = []
    processed_count = 0

    logger.info(f"Processing {len(scraped_contents)} scraped contents")

    for content in scraped_contents:
        if not content.success:
            logger.info(f"Skipping failed scrape: {content.url}")
            continue

        # Validate content quality
        if not validate_content_quality(content):
            logger.info(f"Skipping low-quality content: {content.url}")
            continue

        # Clean the content
        cleaned_content = clean_content(content.content)
        content.content = cleaned_content
        content.content_length = len(cleaned_content)

        # Create chunks
        chunks = chunk_content(content)
        if chunks:
            all_chunks.extend(chunks)
            processed_count += 1
            logger.info(f"Processed {content.url}: {len(chunks)} chunks, {len(cleaned_content)} chars")

    logger.info(f"Content processing complete: {processed_count}/{len(scraped_contents)} sources processed, {len(all_chunks)} total chunks")

    return all_chunks


def summarize_content_stats(chunks: List[ContentChunk]) -> dict:
    """
    Generate summary statistics about processed content.
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'total_words': 0,
            'total_characters': 0,
            'sources': 0,
            'avg_chunk_size': 0,
            'avg_words_per_chunk': 0
        }

    total_words = sum(chunk.word_count for chunk in chunks)
    total_chars = sum(len(chunk.content) for chunk in chunks)
    unique_sources = len(set(chunk.source_url for chunk in chunks))

    return {
        'total_chunks': len(chunks),
        'total_words': total_words,
        'total_characters': total_chars,
        'sources': unique_sources,
        'avg_chunk_size': total_chars // len(chunks),
        'avg_words_per_chunk': total_words // len(chunks)
    }
