"""Text processing utility functions."""

import re
from typing import List


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common HTML entities
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Strip leading/trailing whitespace
    return text.strip()


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text."""
    # Simple sentence splitting (can be improved)
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    # Remove leading/trailing underscores
    return filename.strip('_')


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return "unknown"
