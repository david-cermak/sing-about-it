#!/usr/bin/env python3
"""
Markdown to WAV converter using Kokoro TTS.
Can be used as a library or as a standalone script.
"""

import argparse
import sys
import os
import re
import numpy as np
import soundfile as sf
from typing import Optional, List, Tuple
from kokoro import KPipeline


def convert_markdown_to_wav(
    md_path: str,
    output_path: Optional[str] = None,
    voice: str = 'af_heart',
    speed: float = 1.0,
    lang_code: str = 'a',
    sample_rate: int = 24000
) -> str:
    """
    Convert a Markdown file to WAV audio using Kokoro TTS.

    Args:
        md_path (str): Path to the input Markdown file
        output_path (str, optional): Output WAV file path
        voice (str): Voice to use (default: 'af_heart')
        speed (float): Speaking speed multiplier (default: 1.0)
        lang_code (str): Language code ('a' for American English, etc.)
        sample_rate (int): Audio sample rate (default: 24000)

    Returns:
        str: Path to the generated WAV file

    Raises:
        FileNotFoundError: If markdown file doesn't exist
        Exception: For TTS processing errors
    """
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(md_path)[0] + '.wav'

    try:
        # Read markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Convert markdown to plain text
        plain_text = _markdown_to_text(markdown_content)

        if not plain_text.strip():
            raise ValueError("No text content found in markdown file")

        # Initialize TTS pipeline
        print(f"Initializing TTS pipeline with voice '{voice}'...")
        pipeline = KPipeline(lang_code=lang_code)

        # Process text into sentences
        sentences = _split_into_sentences(plain_text)

        if not sentences:
            raise ValueError("No sentences found in text content")

        print(f"Processing {len(sentences)} sentences...")

        # Generate audio for each sentence
        all_audio_segments = []
        all_generated_text = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            print(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")

            try:
                generator = pipeline(sentence, voice=voice, speed=speed)
                gs, ps, audio = next(generator)

                all_audio_segments.append(audio)
                all_generated_text.append(gs)
                print(f"Generated: {gs[:50]}...")

            except Exception as e:
                print(f"Warning: Failed to process sentence {i+1}: {e}")
                continue

        if not all_audio_segments:
            raise RuntimeError("No audio was generated successfully")

        # Concatenate all audio segments
        print("Concatenating audio segments...")
        complete_audio = np.concatenate(all_audio_segments)

        # Save as WAV file
        print(f"Saving WAV file: {output_path}")
        sf.write(output_path, complete_audio, sample_rate)

        print(f"✓ Successfully generated audio")
        print(f"  Output: {output_path}")
        print(f"  Duration: {len(complete_audio) / sample_rate:.2f} seconds")
        print(f"  Generated text: {len(' '.join(all_generated_text))} characters")

        return output_path

    except Exception as e:
        raise Exception(f"Error converting markdown to WAV: {str(e)}")


def _markdown_to_text(markdown_content: str) -> str:
    """
    Convert markdown content to plain text suitable for TTS.

    Args:
        markdown_content (str): Raw markdown content

    Returns:
        str: Plain text content
    """
    text = markdown_content

    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'`(.*?)`', r'\1', text)        # Inline code
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Code blocks

    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove images
    text = re.sub(r'!\[.*?\]\([^\)]+\)', '', text)

    # Convert bullet points to natural speech
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)

    # Convert numbered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)     # Multiple spaces/tabs to single space

    # Convert paragraph breaks to sentence breaks
    text = text.replace('\n\n', '. ')
    text = text.replace('\n', ' ')

    # Clean up punctuation
    text = re.sub(r'\.{2,}', '.', text)  # Multiple periods
    text = re.sub(r'\s+', ' ', text)     # Multiple spaces

    return text.strip()


def _split_into_sentences(text: str, max_length: int = 500) -> List[str]:
    """
    Split text into sentences suitable for TTS processing.

    Args:
        text (str): Input text
        max_length (int): Maximum sentence length

    Returns:
        List[str]: List of sentences
    """
    # First, split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Further split long sentences at natural break points
    final_sentences = []

    for sentence in sentences:
        if len(sentence) <= max_length:
            final_sentences.append(sentence)
        else:
            # Split long sentences at commas, semicolons, or conjunctions
            parts = re.split(r'[,;]|\s+(?:and|but|or|however|therefore|moreover)\s+', sentence)

            current_part = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if len(current_part + " " + part) <= max_length:
                    if current_part:
                        current_part += " " + part
                    else:
                        current_part = part
                else:
                    if current_part:
                        final_sentences.append(current_part)
                    current_part = part

            if current_part:
                final_sentences.append(current_part)

    # Filter out very short sentences and ensure proper punctuation
    processed_sentences = []
    for sentence in final_sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Skip very short fragments
            # Ensure sentence ends with punctuation
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            processed_sentences.append(sentence)

    return processed_sentences


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description="Convert Markdown file to WAV audio using Kokoro TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available voices:
  af_heart, af_sky, af_nova, af_sarah, af_celeste, af_alloy, am_michael, am_adam,
  am_daniel, am_william, am_christopher, am_matthew, bm_lewis, bf_sarah, bf_emma,
  bf_kimberly, bf_isabella, bf_allison, bf_nova

Language codes:
  'a' => American English 🇺🇸
  'b' => British English 🇬🇧
  'e' => Spanish 🇪🇸
  'f' => French 🇫🇷
  'h' => Hindi 🇮🇳
  'i' => Italian 🇮🇹
  'j' => Japanese 🇯🇵
  'p' => Brazilian Portuguese 🇧🇷
  'z' => Mandarin Chinese 🇨🇳

Examples:
  python convert_md2wav.py document.md
  python convert_md2wav.py document.md --voice af_nova --speed 1.2
  python convert_md2wav.py document.md --output audio.wav
        """
    )

    parser.add_argument(
        'markdown_file',
        help='Path to the input Markdown file'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output WAV file path (default: same name as markdown with .wav extension)'
    )

    parser.add_argument(
        '--voice',
        default='af_heart',
        help='Voice to use for TTS (default: af_heart)'
    )

    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Speaking speed multiplier (default: 1.0)'
    )

    parser.add_argument(
        '--lang-code',
        default='a',
        help='Language code (default: "a" for American English)'
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        default=24000,
        help='Audio sample rate (default: 24000)'
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.markdown_file):
        print(f"Error: Markdown file '{args.markdown_file}' not found!", file=sys.stderr)
        sys.exit(1)

    try:
        # Convert markdown to WAV
        print(f"Converting '{args.markdown_file}' to WAV audio...")

        output_path = convert_markdown_to_wav(
            md_path=args.markdown_file,
            output_path=args.output,
            voice=args.voice,
            speed=args.speed,
            lang_code=args.lang_code,
            sample_rate=args.sample_rate
        )

        print(f"\n✓ Conversion complete!")
        print(f"  Audio saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
