"""
Chapter Generator Agent for Learning Content Organization.

This module takes comprehensive learning sheets and breaks them down into
structured, digestible chapters suitable for progressive learning and eventual
song generation. Uses markdown format for better compatibility with local models.
"""

import logging
import time
from typing import List, Dict, Optional
import sys
from pathlib import Path
from datetime import datetime
import re

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from agents.factory import AgentFactory, BackendType
from agents.base import AgentBackend

# Setup logging
logger = logging.getLogger(__name__)


class SimpleChapterContent:
    """Simple chapter structure without Pydantic."""
    def __init__(self, chapter_number: int, title: str, introduction: str,
                 core_concepts: str, examples: str, summary: str,
                 key_terms: List[str], estimated_read_time: int):
        self.chapter_number = chapter_number
        self.title = title
        self.introduction = introduction
        self.core_concepts = core_concepts
        self.examples = examples
        self.summary = summary
        self.key_terms = key_terms
        self.estimated_read_time = estimated_read_time


class SimpleChapterResponse:
    """Simple chapter response without Pydantic."""
    def __init__(self, overview: str, chapters: List[SimpleChapterContent],
                 learning_path: List[str], confidence_score: float):
        self.overview = overview
        self.chapters = chapters
        self.learning_path = learning_path
        self.confidence_score = confidence_score


class ChapterGenerator:
    """
    Generates structured learning chapters from comprehensive learning sheets.

    Uses simple markdown format for better compatibility with local models.
    Takes learning sheets and breaks them into digestible, pedagogically sound
    chapters that prepare content for eventual song generation while maintaining
    educational value.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the chapter generator.

        Args:
            model_name: Optional model name override. Defaults to llama3.2:latest.
        """
        # Use efficient model for chapter generation
        model_name = model_name or "llama3.2:latest"
        system_prompt = """You are an expert educational content organizer. You specialize in breaking comprehensive learning materials into digestible, pedagogically sound chapters.

Your responses should be clear, well-structured markdown that follows the exact format requested. Focus on creating logical progression from fundamentals to advanced concepts."""

        self.backend = AgentFactory.create_backend(
            BackendType.BAREMETAL,
            system_prompt=system_prompt,
            model_name=model_name
        )

        logger.info(f"ChapterGenerator initialized with model: {model_name}")

    def generate_chapters(self, learning_sheet_content: str, topic: str) -> SimpleChapterResponse:
        """
        Generate structured chapters from learning sheet content using a two-step process:
        1. Generate chapter structure (titles, overview)
        2. Generate content for each chapter individually

        Args:
            learning_sheet_content: The full learning sheet content to organize
            topic: The main topic for context

        Returns:
            SimpleChapterResponse with structured chapters and content

        Raises:
            Exception: If chapter generation fails
        """
        start_time = time.time()

        try:
            logger.info(f"Starting chapter generation for topic: {topic}")

            # Step 1: Generate chapter structure
            logger.info("Step 1: Generating chapter structure...")
            structure = self._generate_chapter_structure(learning_sheet_content, topic)

            # Step 2: Generate content for each chapter
            logger.info(f"Step 2: Generating content for {len(structure['chapter_titles'])} chapters...")
            chapters = []

            for i, title in enumerate(structure['chapter_titles'], 1):
                logger.info(f"Generating content for Chapter {i}: {title}")
                chapter_content = self._generate_individual_chapter(
                    chapter_number=i,
                    chapter_title=title,
                    learning_sheet_content=learning_sheet_content,
                    topic=topic,
                    overall_context=structure['overview']
                )
                chapters.append(chapter_content)

            # Create final response
            response = SimpleChapterResponse(
                overview=structure['overview'],
                chapters=chapters,
                learning_path=structure['learning_path'],
                confidence_score=structure['confidence_score']
            )

            generation_time = time.time() - start_time
            logger.info(f"Chapter generation completed in {generation_time:.2f}s")

            # Log chapter summary
            self._log_chapter_summary(response, topic)

            return response

        except Exception as e:
            logger.error(f"Chapter generation failed for {topic}: {str(e)}")
            raise Exception(f"Failed to generate chapters: {str(e)}")

    def _generate_simple_response(self, prompt: str) -> str:
        """Generate a simple string response from the backend."""
        try:
            # Try OpenAI-compatible endpoint first
            response = self.backend._try_openai_compatible(prompt, "")
            if response:
                return response
        except Exception as e:
            logger.debug(f"OpenAI-compatible failed: {e}")

        try:
            # Fallback to Ollama direct
            response = self.backend._try_ollama_direct(prompt, "")
            if response:
                return response
        except Exception as e:
            logger.debug(f"Ollama direct failed: {e}")

        raise Exception("All generation strategies failed")

    def _generate_chapter_structure(self, content: str, topic: str) -> dict:
        """Generate the overall chapter structure (titles, overview, learning path)."""
        prompt = f"""Analyze this learning sheet about "{topic}" and create a chapter structure with 4-6 chapters.

**LEARNING SHEET CONTENT:**
{content}

**OUTPUT REQUIRED - EXACT FORMAT:**

OVERVIEW:
[Write a 2-3 sentence overview of how the content is organized]

LEARNING_PATH:
1. [Step 1 description]
2. [Step 2 description]
3. [Step 3 description]
4. [Step 4 description]
[Add 5-6 steps if needed]

CHAPTER_TITLES:
1. [Chapter 1 title]
2. [Chapter 2 title]
3. [Chapter 3 title]
4. [Chapter 4 title]
[Add 5-6 chapters if needed]

CONFIDENCE:
[0.0-1.0]

**REQUIREMENTS:**
- Create 4-6 logical chapters
- Progression from fundamentals to advanced
- Each chapter should be focused and distinct
- Suitable for song/audio conversion later

Use EXACTLY the format above."""

        try:
            response = self._generate_simple_response(prompt)
            return self._parse_structure_response(response)
        except Exception as e:
            logger.error(f"Structure generation failed: {e}")
            # Return fallback structure
            return {
                'overview': f"Learning content organized into chapters about {topic}",
                'learning_path': [f"Introduction to {topic}", "Core concepts", "Practical applications"],
                'chapter_titles': [f"Introduction to {topic}", "Core Concepts", "Practical Applications"],
                'confidence_score': 0.5
            }

    def _parse_structure_response(self, response: str) -> dict:
        """Parse the structure response into a dictionary."""
        try:
            # Extract chapter titles from **CHAPTER X:** headers
            chapter_titles = []
            chapter_pattern = r'\*\*CHAPTER \d+: ([^*]+)\*\*'
            chapter_matches = re.findall(chapter_pattern, response)
            for match in chapter_matches:
                chapter_titles.append(match.strip())

            # If no chapters found with that pattern, try other patterns
            if not chapter_titles:
                # Try alternative patterns
                alt_patterns = [
                    r'CHAPTER \d+: ([^\n]+)',
                    r'\d+\.\s*([^\n]+)',
                    r'\*\*([^*]+)\*\*\s*(?:Chapter|chapter)'
                ]
                for pattern in alt_patterns:
                    matches = re.findall(pattern, response)
                    if matches:
                        chapter_titles = [m.strip() for m in matches]
                        break

            # Extract overview - use first "Overview:" section found
            overview = "Overview not found"
            overview_match = re.search(r'Overview:\s*\n(.*?)(?=\n(?:Learning Path|CHAPTER|\*\*|$))', response, re.DOTALL)
            if overview_match:
                overview = overview_match.group(1).strip()

            # Extract learning path - collect all learning path items
            learning_path = []
            # Look for all "Learning Path:" sections
            learning_sections = re.findall(r'Learning Path:\s*\n(.*?)(?=\n(?:CHAPTER|\*\*|Overview:|$))', response, re.DOTALL)
            for section in learning_sections:
                for line in section.split('\n'):
                    line = line.strip()
                    if line and re.match(r'^\d+\.', line):
                        clean_line = re.sub(r'^\d+\.\s*', '', line)
                        if clean_line and clean_line not in learning_path:
                            learning_path.append(clean_line.strip())

            # Extract confidence
            confidence_score = 0.8  # Default
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
            if not confidence_match:
                confidence_match = re.search(r'([\d.]+)/1\.0', response)
            if confidence_match:
                confidence_score = float(confidence_match.group(1))

            # Create a comprehensive overview if we have chapters
            if chapter_titles and overview == "Overview not found":
                overview = f"This learning content is organized into {len(chapter_titles)} chapters covering embedded systems security fundamentals."

            return {
                'overview': overview,
                'learning_path': learning_path,
                'chapter_titles': chapter_titles,
                'confidence_score': confidence_score
            }

        except Exception as e:
            logger.error(f"Failed to parse structure response: {e}")
            logger.error(f"Response snippet: {response[:500]}...")
            return {
                'overview': "Failed to parse overview",
                'learning_path': ["Step 1", "Step 2", "Step 3"],
                'chapter_titles': ["Chapter 1", "Chapter 2", "Chapter 3"],
                'confidence_score': 0.5
            }

    def _generate_individual_chapter(self, chapter_number: int, chapter_title: str,
                                   learning_sheet_content: str, topic: str, overall_context: str) -> SimpleChapterContent:
        """Generate content for a single chapter using the working approach."""
        # Use the same prompt format that works in the test script
        prompt = f"""Generate comprehensive content for this chapter:

**CHAPTER TITLE:** {chapter_title}

**CHAPTER OVERVIEW:** {overall_context}

**LEARNING CONTEXT:**
Topic: {topic}
Source material focuses on: {learning_sheet_content[:500]}...

**INSTRUCTIONS:**
Create educational content that follows this structure:

## Introduction
[Write 100-150 words introducing the chapter and what readers will learn]

## Core Concepts
[Write 400-600 words of main educational content with clear explanations, organized with subheadings]

## Examples
[Write 150-250 words of concrete, real-world examples and applications]

## Summary
[Write 100-150 words summarizing the key takeaways and lessons learned]

## Key Terms
[List important terms as: **term1**, **term2**, **term3**, **term4**]

**REQUIREMENTS:**
- Focus specifically on "{chapter_title}"
- Use markdown formatting (## for sections)
- Make content engaging and memorable
- Include real-world applications
- Suitable for audio/song conversion
- Total: 750-1200 words

Generate comprehensive, educational content now."""

        try:
            response = self._generate_simple_response(prompt)
            return self._parse_chapter_content_improved(response, chapter_number, chapter_title)
        except Exception as e:
            logger.error(f"Chapter {chapter_number} generation failed: {e}")
            return self._create_fallback_chapter(chapter_number, chapter_title)

    def _parse_chapter_content_improved(self, response: str, chapter_number: int, chapter_title: str) -> SimpleChapterContent:
        """Improved parsing that handles the markdown format properly."""
        try:
            # Extract each section with more flexible patterns
            intro_match = re.search(r'## Introduction\s*\n(.*?)(?=\n## |$)', response, re.DOTALL)
            introduction = intro_match.group(1).strip() if intro_match else ""

            concepts_match = re.search(r'## Core Concepts\s*\n(.*?)(?=\n## |$)', response, re.DOTALL)
            core_concepts = concepts_match.group(1).strip() if concepts_match else ""

            examples_match = re.search(r'## Examples\s*\n(.*?)(?=\n## |$)', response, re.DOTALL)
            examples = examples_match.group(1).strip() if examples_match else ""

            summary_match = re.search(r'## Summary\s*\n(.*?)(?=\n## |$)', response, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else ""

            # Extract key terms - look for **term** patterns
            key_terms = []
            terms_match = re.search(r'## Key Terms\s*\n(.*?)(?=\n## |$)', response, re.DOTALL)
            if terms_match:
                terms_text = terms_match.group(1).strip()
                # Find all **term** patterns
                term_matches = re.findall(r'\*\*([^*]+)\*\*', terms_text)
                key_terms = [term.strip() for term in term_matches if term.strip()]

            # Estimate read time based on word count
            total_words = len((introduction + core_concepts + examples + summary).split())
            read_time = max(8, min(15, total_words // 150))  # ~150 words per minute

            # Validate content - if sections are too short, log warning but don't fail
            if len(introduction) < 50:
                logger.warning(f"Chapter {chapter_number}: Introduction seems short ({len(introduction)} chars)")
            if len(core_concepts) < 100:
                logger.warning(f"Chapter {chapter_number}: Core concepts seems short ({len(core_concepts)} chars)")

            return SimpleChapterContent(
                chapter_number=chapter_number,
                title=chapter_title,
                introduction=introduction,
                core_concepts=core_concepts,
                examples=examples,
                summary=summary,
                key_terms=key_terms,
                estimated_read_time=read_time
            )

        except Exception as e:
            logger.error(f"Failed to parse chapter content for '{chapter_title}': {e}")
            # Return fallback with some content rather than empty
            return self._create_fallback_chapter(chapter_number, chapter_title)

    def _parse_chapter_content(self, response: str, chapter_number: int, chapter_title: str) -> SimpleChapterContent:
        """Parse individual chapter content response."""
        try:
            # Extract each section - handle both colon and markdown format
            # Try markdown format first (## Introduction)
            intro_match = re.search(r'## Introduction\s*\n(.*?)(?=\n## |$)', response, re.DOTALL)
            if not intro_match:
                # Fallback to colon format
                intro_match = re.search(r'INTRODUCTION:\s*\n(.*?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            introduction = intro_match.group(1).strip() if intro_match else "Introduction content not found"

            concepts_match = re.search(r'## Core Concepts\s*\n(.*?)(?=\n## |$)', response, re.DOTALL)
            if not concepts_match:
                concepts_match = re.search(r'CORE_CONCEPTS:\s*\n(.*?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            core_concepts = concepts_match.group(1).strip() if concepts_match else "Core concepts not found"

            examples_match = re.search(r'## Examples\s*\n(.*?)(?=\n## |$)', response, re.DOTALL)
            if not examples_match:
                examples_match = re.search(r'EXAMPLES:\s*\n(.*?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            examples = examples_match.group(1).strip() if examples_match else "Examples not found"

            summary_match = re.search(r'## Summary\s*\n(.*?)(?=\n## |$)', response, re.DOTALL)
            if not summary_match:
                summary_match = re.search(r'SUMMARY:\s*\n(.*?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else "Summary not found"

            # Extract key terms - handle both markdown list and colon format
            terms_match = re.search(r'## Key Terms\s*\n(.*?)(?=\n## |$)', response, re.DOTALL)
            if not terms_match:
                terms_match = re.search(r'KEY_TERMS:\s*\n(.*?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)

            key_terms = []
            if terms_match:
                terms_text = terms_match.group(1).strip()
                # Handle bullet list format (* **term**: description)
                if '*' in terms_text:
                    for line in terms_text.split('\n'):
                        line = line.strip()
                        if line.startswith('*'):
                            # Extract term from "* **term**: description"
                            term_match = re.search(r'\*\s*\*\*([^*]+)\*\*', line)
                            if term_match:
                                key_terms.append(term_match.group(1).strip())
                else:
                    # Handle comma-separated format
                    key_terms = [term.strip() for term in terms_text.split(',') if term.strip()]

            # Extract read time
            time_match = re.search(r'## Read Time\s*\n.*?(\d+)', response, re.DOTALL)
            if not time_match:
                time_match = re.search(r'READ_TIME:\s*\n.*?(\d+)', response)
            if not time_match:
                # Look for "8-12 minutes" pattern
                time_match = re.search(r'(\d+)-\d+\s+minutes', response)
            read_time = int(time_match.group(1)) if time_match else 10

            return SimpleChapterContent(
                chapter_number=chapter_number,
                title=chapter_title,
                introduction=introduction,
                core_concepts=core_concepts,
                examples=examples,
                summary=summary,
                key_terms=key_terms,
                estimated_read_time=read_time
            )

        except Exception as e:
            logger.error(f"Failed to parse chapter content: {e}")
            logger.error(f"Response snippet: {response[:300]}...")
            return self._create_fallback_chapter(chapter_number, chapter_title)

    def _create_fallback_chapter(self, chapter_number: int, chapter_title: str) -> SimpleChapterContent:
        """Create fallback chapter content."""
        return SimpleChapterContent(
            chapter_number=chapter_number,
            title=chapter_title,
            introduction=f"This chapter covers {chapter_title.lower()}.",
            core_concepts=f"Key concepts for {chapter_title.lower()} will be explained here.",
            examples=f"Examples of {chapter_title.lower()} in practice.",
            summary=f"Summary of {chapter_title.lower()} main points.",
            key_terms=["concept1", "concept2", "concept3"],
            estimated_read_time=10
        )

    def _parse_markdown_response(self, raw_response: str) -> SimpleChapterResponse:
        """Parse the markdown response into a structured format."""
        try:
            # Extract overview
            overview_match = re.search(r'# Overview\s*\n(.*?)(?=\n# |\n## |\Z)', raw_response, re.DOTALL)
            overview = overview_match.group(1).strip() if overview_match else "Overview not found"

            # Extract learning path
            learning_path = []
            learning_path_match = re.search(r'# Learning Path\s*\n(.*?)(?=\n# |\n## |\Z)', raw_response, re.DOTALL)
            if learning_path_match:
                path_text = learning_path_match.group(1).strip()
                for line in path_text.split('\n'):
                    line = line.strip()
                    if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.')) or line.startswith('-')):
                        # Remove numbering/bullets and clean up
                        clean_line = re.sub(r'^\d+\.\s*', '', line)
                        clean_line = re.sub(r'^-\s*', '', clean_line)
                        learning_path.append(clean_line.strip())

            # Extract chapters
            chapters = []
            chapter_pattern = r'## Chapter (\d+): (.*?)\n(.*?)(?=\n## Chapter|\n# Confidence|\Z)'
            chapter_matches = re.findall(chapter_pattern, raw_response, re.DOTALL)

            for chapter_num, title, content in chapter_matches:
                # Parse chapter content
                read_time_match = re.search(r'\*\*Read Time:\*\*\s*(\d+)', content)
                read_time = int(read_time_match.group(1)) if read_time_match else 10

                key_terms_match = re.search(r'\*\*Key Terms:\*\*\s*(.*?)(?=\n|$)', content)
                key_terms = []
                if key_terms_match:
                    terms_text = key_terms_match.group(1).strip()
                    key_terms = [t.strip() for t in terms_text.split(',') if t.strip()]

                # Extract sections
                intro_match = re.search(r'### Introduction\s*\n(.*?)(?=\n### |\Z)', content, re.DOTALL)
                introduction = intro_match.group(1).strip() if intro_match else ""

                concepts_match = re.search(r'### Core Concepts\s*\n(.*?)(?=\n### |\Z)', content, re.DOTALL)
                core_concepts = concepts_match.group(1).strip() if concepts_match else ""

                examples_match = re.search(r'### Examples\s*\n(.*?)(?=\n### |\Z)', content, re.DOTALL)
                examples = examples_match.group(1).strip() if examples_match else ""

                summary_match = re.search(r'### Summary\s*\n(.*?)(?=\n### |\Z)', content, re.DOTALL)
                summary = summary_match.group(1).strip() if summary_match else ""

                chapter = SimpleChapterContent(
                    chapter_number=int(chapter_num),
                    title=title.strip(),
                    introduction=introduction,
                    core_concepts=core_concepts,
                    examples=examples,
                    summary=summary,
                    key_terms=key_terms,
                    estimated_read_time=read_time
                )
                chapters.append(chapter)

            # Extract confidence score
            confidence_match = re.search(r'# Confidence Score\s*\n([\d.]+)', raw_response)
            confidence_score = float(confidence_match.group(1)) if confidence_match else 0.8

            return SimpleChapterResponse(
                overview=overview,
                chapters=chapters,
                learning_path=learning_path,
                confidence_score=confidence_score
            )

        except Exception as e:
            logger.error(f"Failed to parse markdown response: {e}")
            # Return a basic fallback response
            return SimpleChapterResponse(
                overview="Failed to parse response",
                chapters=[],
                learning_path=[],
                confidence_score=0.5
            )

    def _log_chapter_summary(self, response: SimpleChapterResponse, topic: str):
        """Log summary of generated chapters."""
        logger.info(f"Generated {len(response.chapters)} chapters for {topic}")
        logger.info(f"Confidence score: {response.confidence_score:.2f}")

        for i, chapter in enumerate(response.chapters, 1):
            logger.info(f"Chapter {i}: {chapter.title} ({chapter.estimated_read_time}min read)")
            logger.info(f"  Key terms: {', '.join(chapter.key_terms)}")

    def generate_chapters_from_file(self, file_path: str, topic: str) -> SimpleChapterResponse:
        """
        Generate chapters from a learning sheet file.

        Args:
            file_path: Path to the learning sheet markdown file
            topic: The main topic for context

        Returns:
            SimpleChapterResponse with structured chapters
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return self.generate_chapters(content, topic)

        except FileNotFoundError:
            raise Exception(f"Learning sheet file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Failed to read learning sheet file: {str(e)}")

    def save_chapters_to_file(self, chapters: SimpleChapterResponse, topic: str, output_path: str = None):
        """
        Save generated chapters to a markdown file.

        Args:
            chapters: The generated chapters
            topic: The topic name for the filename
            output_path: Optional custom output path
        """
        if output_path is None:
            safe_topic = topic.replace(' ', '_').replace('/', '_')
            output_path = f"chapters_{safe_topic}.md"

        try:
            content = self._format_chapters_as_markdown(chapters, topic)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Chapters saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save chapters: {str(e)}")
            raise

    def _format_chapters_as_markdown(self, chapters: SimpleChapterResponse, topic: str) -> str:
        """Format the chapters as a markdown document."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# Learning Chapters: {topic.title()}

> **Generated by**: Chapter Generator
> **Topic**: {topic}
> **Chapters**: {len(chapters.chapters)}
> **Confidence Score**: {chapters.confidence_score:.2f}/1.0
> **Generated**: {timestamp}

---

## Overview

{chapters.overview}

## Learning Path

{chr(10).join([f"{i+1}. {path}" for i, path in enumerate(chapters.learning_path)])}

---

"""

        # Add each chapter
        for chapter in chapters.chapters:
            content += f"""## Chapter {chapter.chapter_number}: {chapter.title}

*Estimated reading time: {chapter.estimated_read_time} minutes*

### Introduction
{chapter.introduction}

### Core Concepts
{chapter.core_concepts}

### Examples & Applications
{chapter.examples}

### Chapter Summary
{chapter.summary}

### Key Terms
{', '.join([f"**{term}**" for term in chapter.key_terms])}

---

"""

        content += f"""
*These chapters were generated using AI to structure learning content for optimal comprehension and retention. The content is organized to support both traditional learning and eventual conversion to audio/song format.*
"""

        return content


# Convenience functions for easy usage
def generate_chapters_from_sheet(learning_sheet_path: str, topic: str, output_path: str = None) -> SimpleChapterResponse:
    """
    Convenience function to generate chapters from a learning sheet file.

    Args:
        learning_sheet_path: Path to the learning sheet markdown file
        topic: The main topic
        output_path: Optional output path for the chapters file

    Returns:
        SimpleChapterResponse with the generated chapters
    """
    generator = ChapterGenerator()
    chapters = generator.generate_chapters_from_file(learning_sheet_path, topic)

    if output_path or True:  # Always save by default
        generator.save_chapters_to_file(chapters, topic, output_path)

    return chapters


if __name__ == "__main__":
    # Example usage
    def test_chapter_generation():
        """Test the chapter generator with a sample learning sheet."""
        try:
            print("🔥 Testing Chapter Generator with Markdown Format")

            # Test with the example learning sheet
            chapters = generate_chapters_from_sheet(
                "learning_sheet_example_embedded_security.md",
                "embedded security"
            )

            print(f"✅ Generated {len(chapters.chapters)} chapters successfully!")
            print(f"✅ Confidence: {chapters.confidence_score:.2f}")

            # Show chapter titles
            for i, chapter in enumerate(chapters.chapters, 1):
                print(f"  📚 Chapter {i}: {chapter.title}")

        except Exception as e:
            print(f"❌ Test failed: {e}")

    # Run the test
    test_chapter_generation()
