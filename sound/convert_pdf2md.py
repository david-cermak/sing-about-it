#!/usr/bin/env python3
"""
PDF to Markdown converter.
Can be used as a library or as a standalone script.
"""

import pymupdf  # PyMuPDF
import argparse
import sys
import os
import re
from typing import Optional


def convert_pdf_to_markdown(pdf_path: str, include_page_numbers: bool = True) -> str:
    """
    Convert a PDF file to Markdown format.

    Args:
        pdf_path (str): Path to the PDF file
        include_page_numbers (bool): Whether to include page number headers

    Returns:
        str: Markdown formatted content

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: For other PDF processing errors
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Open the PDF document
        doc = pymupdf.open(pdf_path)

        if doc.page_count == 0:
            return "# Empty Document\n\nThis PDF contains no pages."

        markdown_content = []

        # Add document title (from filename)
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        markdown_content.append(f"# {filename.replace('_', ' ').replace('-', ' ').title()}\n")

        # Process each page
        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Extract text from the page
            text = page.get_text()

            if text.strip():  # Only process pages with content
                if include_page_numbers and doc.page_count > 1:
                    markdown_content.append(f"\n## Page {page_num + 1}\n")

                # Clean and format the text
                formatted_text = _format_text_as_markdown(text)
                markdown_content.append(formatted_text)
                markdown_content.append("\n")

        doc.close()

        # Join all content and clean up extra newlines
        full_markdown = ''.join(markdown_content)
        full_markdown = re.sub(r'\n{3,}', '\n\n', full_markdown)  # Max 2 consecutive newlines

        return full_markdown.strip() + '\n'

    except Exception as e:
        raise Exception(f"Error processing PDF '{pdf_path}': {str(e)}")


def _format_text_as_markdown(text: str) -> str:
    """
    Format extracted PDF text as markdown, attempting to preserve structure.

    Args:
        text (str): Raw text from PDF

    Returns:
        str: Formatted markdown text
    """
    if not text.strip():
        return ""

    lines = text.split('\n')
    formatted_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            formatted_lines.append('')
            continue

        # Detect potential headers (lines that are short and appear to be titles)
        if _looks_like_header(line):
            # Make it a markdown header
            formatted_lines.append(f"### {line}")
        else:
            # Regular paragraph text
            formatted_lines.append(line)

    # Join lines and handle paragraphs
    markdown_text = '\n'.join(formatted_lines)

    # Convert double+ newlines to paragraph breaks
    markdown_text = re.sub(r'\n\n+', '\n\n', markdown_text)

    # Convert single newlines within paragraphs to spaces (for better readability)
    # But preserve intentional breaks (like bullet points or numbered lists)
    paragraphs = markdown_text.split('\n\n')
    formatted_paragraphs = []

    for paragraph in paragraphs:
        if paragraph.strip():
            # Check if it's a list or structured content
            if _is_list_content(paragraph):
                formatted_paragraphs.append(paragraph)
            else:
                # Join lines within paragraph with spaces
                joined_paragraph = ' '.join(line.strip() for line in paragraph.split('\n') if line.strip())
                formatted_paragraphs.append(joined_paragraph)

    return '\n\n'.join(formatted_paragraphs)


def _looks_like_header(line: str) -> bool:
    """
    Determine if a line looks like a header/title.

    Args:
        line (str): Text line to check

    Returns:
        bool: True if line appears to be a header
    """
    # Skip very short lines (< 3 chars) or very long lines (> 100 chars)
    if len(line) < 3 or len(line) > 100:
        return False

    # Headers often don't end with punctuation
    if line.endswith('.'):
        return False

    # Headers are often in title case or all caps
    words = line.split()
    if len(words) > 1:
        title_case_words = sum(1 for word in words if word and (word[0].isupper() or word.isupper()))
        if title_case_words / len(words) > 0.5:
            return True

    # Check for common header patterns
    header_patterns = [
        r'^[IVX]+\.',  # Roman numerals
        r'^\d+\.',     # Numbered sections
        r'^(Chapter|Section|Part)\s+\d+',  # Chapter/Section headers
        r'^[A-Z][A-Z\s]+$',  # All caps
    ]

    for pattern in header_patterns:
        if re.match(pattern, line):
            return True

    return False


def _is_list_content(paragraph: str) -> bool:
    """
    Check if paragraph contains list-like content that should preserve line breaks.

    Args:
        paragraph (str): Paragraph text to check

    Returns:
        bool: True if content appears to be a list
    """
    lines = [line.strip() for line in paragraph.split('\n') if line.strip()]

    if len(lines) < 2:
        return False

    # Check for bullet points or numbered lists
    list_patterns = [
        r'^\s*[-•*]\s+',  # Bullet points
        r'^\s*\d+\.\s+',  # Numbered lists
        r'^\s*[a-zA-Z]\.\s+',  # Lettered lists
        r'^\s*\([a-zA-Z0-9]+\)\s+',  # Parenthetical lists
    ]

    matching_lines = 0
    for line in lines:
        for pattern in list_patterns:
            if re.match(pattern, line):
                matching_lines += 1
                break

    # If more than half the lines look like list items
    return matching_lines / len(lines) > 0.5


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description="Convert PDF file to Markdown format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_pdf2md.py document.pdf
  python convert_pdf2md.py document.pdf --output document.md
  python convert_pdf2md.py document.pdf --no-page-numbers
        """
    )

    parser.add_argument(
        'pdf_file',
        help='Path to the input PDF file'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output markdown file path (default: same name as PDF with .md extension)'
    )

    parser.add_argument(
        '--no-page-numbers',
        action='store_true',
        help='Don\'t include page number headers in the output'
    )

    parser.add_argument(
        '--stdout',
        action='store_true',
        help='Print markdown to stdout instead of saving to file'
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file '{args.pdf_file}' not found!", file=sys.stderr)
        sys.exit(1)

    try:
        # Convert PDF to markdown
        print(f"Converting '{args.pdf_file}' to Markdown...", file=sys.stderr)

        include_page_numbers = not args.no_page_numbers
        markdown_content = convert_pdf_to_markdown(args.pdf_file, include_page_numbers)

        if args.stdout:
            # Print to stdout
            print(markdown_content)
        else:
            # Save to file
            if args.output:
                output_path = args.output
            else:
                # Default: replace .pdf extension with .md
                output_path = os.path.splitext(args.pdf_file)[0] + '.md'

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"✓ Conversion complete!", file=sys.stderr)
            print(f"  Output: {output_path}", file=sys.stderr)
            print(f"  Size: {len(markdown_content)} characters", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
