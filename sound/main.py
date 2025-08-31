#!/usr/bin/env python3
"""
Main script for PDF to audio conversion pipeline.
Supports PDF to Markdown, PDF to WAV, and full PDF to MP3 conversion.
"""

import argparse
import sys
import os
import tempfile
from convert_pdf2md import convert_pdf_to_markdown
from convert_md2wav import convert_markdown_to_wav
from convert_wav2mp3 import convert_wav_to_mp3


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF through multiple processing stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py document.pdf                    # Full pipeline: PDF → MP3
  python main.py document.pdf --voice af_nova --speed 1.2 --bitrate 320k
  python main.py document.pdf --markdown-only   # PDF → Markdown only
  python main.py document.pdf --wav-only        # PDF → WAV only
        """
    )

    parser.add_argument(
        'pdf_file',
        help='Path to the input PDF file'
    )

    parser.add_argument(
        '--markdown-only',
        action='store_true',
        help='Convert PDF to Markdown only'
    )

    parser.add_argument(
        '--wav-only',
        action='store_true',
        help='Convert PDF to WAV audio (PDF → Markdown → WAV)'
    )

    parser.add_argument(
        '--output',
        '-o',
        help='Output file path (optional, defaults to PDF name with appropriate extension)'
    )

    parser.add_argument(
        '--voice',
        default='af_heart',
        help='Voice to use for TTS (only for --wav-only, default: af_heart)'
    )

    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Speaking speed multiplier (only for --wav-only, default: 1.0)'
    )

    parser.add_argument(
        '--bitrate', '-b',
        default='192k',
        help='MP3 bitrate (default: 192k). Options: 128k, 192k, 256k, 320k'
    )

    parser.add_argument(
        '--keep-intermediate',
        action='store_true',
        help='Keep intermediate files (markdown and WAV files)'
    )

    parser.add_argument(
        '--keep-markdown',
        action='store_true',
        help='Keep intermediate markdown file (backward compatibility)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file '{args.pdf_file}' not found!", file=sys.stderr)
        sys.exit(1)

    if not args.pdf_file.lower().endswith('.pdf'):
        print(f"Warning: '{args.pdf_file}' doesn't have .pdf extension", file=sys.stderr)

    # Check for conflicting options
    if args.markdown_only and args.wav_only:
        print("Error: Cannot use both --markdown-only and --wav-only options", file=sys.stderr)
        sys.exit(1)

    if args.markdown_only:
        try:
            # Determine output file path
            if args.output:
                output_path = args.output
            else:
                # Default: replace .pdf with .md
                output_path = os.path.splitext(args.pdf_file)[0] + '.md'

            print(f"Converting PDF to Markdown...")
            print(f"Input: {args.pdf_file}")
            print(f"Output: {output_path}")

            # Convert PDF to Markdown
            markdown_content = convert_pdf_to_markdown(args.pdf_file)

            # Write to output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"✓ Successfully converted PDF to Markdown")
            print(f"  Output saved to: {output_path}")
            print(f"  Content length: {len(markdown_content)} characters")

        except Exception as e:
            print(f"Error during PDF to Markdown conversion: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.wav_only:
        try:
            # Determine output file path
            if args.output:
                output_path = args.output
                if not output_path.lower().endswith('.wav'):
                    output_path += '.wav'
            else:
                # Default: replace .pdf with .wav
                output_path = os.path.splitext(args.pdf_file)[0] + '.wav'

            print(f"Converting PDF to WAV audio...")
            print(f"Input: {args.pdf_file}")
            print(f"Output: {output_path}")
            print(f"Voice: {args.voice}")
            print(f"Speed: {args.speed}x")

            # Step 1: Convert PDF to Markdown
            print("\n[1/2] Converting PDF to Markdown...")
            markdown_content = convert_pdf_to_markdown(args.pdf_file)

            # Create temporary markdown file or use permanent one
            keep_files = args.keep_intermediate or args.keep_markdown
            if keep_files:
                # Create permanent markdown file
                md_path = os.path.splitext(args.pdf_file)[0] + '.md'
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"  ✓ Markdown saved to: {md_path}")
            else:
                # Create temporary markdown file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                    f.write(markdown_content)
                    md_path = f.name

            # Step 2: Convert Markdown to WAV
            print(f"\n[2/2] Converting Markdown to WAV audio...")
            wav_path = convert_markdown_to_wav(
                md_path=md_path,
                output_path=output_path,
                voice=args.voice,
                speed=args.speed
            )

            # Clean up temporary markdown file if not keeping it
            if not keep_files:
                try:
                    os.unlink(md_path)
                    print(f"  ✓ Cleaned up temporary markdown file")
                except Exception:
                    pass  # Ignore cleanup errors

            print(f"\n✅ PDF to WAV conversion complete!")
            print(f"  Final output: {wav_path}")

        except Exception as e:
            print(f"Error during PDF to WAV conversion: {e}", file=sys.stderr)
            # Cleanup on error
            keep_files = args.keep_intermediate or args.keep_markdown
            if 'md_path' in locals() and not keep_files:
                try:
                    os.unlink(md_path)
                except Exception:
                    pass
            sys.exit(1)

    else:
        # Full pipeline: PDF → Markdown → WAV → MP3
        try:
            # Determine output file path
            if args.output:
                output_path = args.output
                if not output_path.lower().endswith('.mp3'):
                    output_path += '.mp3'
            else:
                # Default: replace .pdf with .mp3
                output_path = os.path.splitext(args.pdf_file)[0] + '.mp3'

            print(f"🎵 Converting PDF to MP3 audio (Full Pipeline)")
            print(f"Input: {args.pdf_file}")
            print(f"Output: {output_path}")
            print(f"Voice: {args.voice}")
            print(f"Speed: {args.speed}x")
            print(f"Bitrate: {args.bitrate}")
            print()

            # Step 1: Convert PDF to Markdown
            print("[1/3] Converting PDF to Markdown...")
            markdown_content = convert_pdf_to_markdown(args.pdf_file)

            # Create markdown file (temporary or permanent)
            keep_files = args.keep_intermediate or args.keep_markdown
            if keep_files:
                # Create permanent markdown file
                md_path = os.path.splitext(args.pdf_file)[0] + '.md'
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"  ✓ Markdown saved to: {md_path}")
            else:
                # Create temporary markdown file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                    f.write(markdown_content)
                    md_path = f.name
                print(f"  ✓ Markdown content processed ({len(markdown_content)} characters)")

            # Step 2: Convert Markdown to WAV
            print(f"\n[2/3] Converting Markdown to WAV audio...")

            # Determine WAV path
            if keep_files:
                wav_path = os.path.splitext(args.pdf_file)[0] + '.wav'
            else:
                # Create temporary WAV file
                wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
                os.close(wav_fd)  # Close the file descriptor, but keep the path

            # Convert to WAV
            final_wav_path = convert_markdown_to_wav(
                md_path=md_path,
                output_path=wav_path,
                voice=args.voice,
                speed=args.speed
            )

            if keep_files:
                print(f"  ✓ WAV audio saved to: {final_wav_path}")
            else:
                print(f"  ✓ WAV audio generated")

            # Step 3: Convert WAV to MP3
            print(f"\n[3/3] Converting WAV to MP3...")
            final_mp3_path = convert_wav_to_mp3(
                wav_path=final_wav_path,
                output_path=output_path,
                bitrate=args.bitrate,
                verbose=False  # We'll handle our own output
            )

            print(f"  ✓ MP3 conversion complete")

            # Clean up temporary files if not keeping them
            cleanup_files = []
            if not keep_files:
                cleanup_files = [md_path, final_wav_path]

                for temp_file in cleanup_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception:
                        pass  # Ignore cleanup errors

                if cleanup_files:
                    print(f"  ✓ Cleaned up temporary files")

            # Final success message
            print(f"\n🎉 Full pipeline conversion complete!")
            print(f"  📄 Input PDF: {args.pdf_file}")
            print(f"  🎵 Output MP3: {final_mp3_path}")

            # Show file info
            if os.path.exists(final_mp3_path):
                mp3_size = os.path.getsize(final_mp3_path) / (1024 * 1024)  # MB
                print(f"  📊 File size: {mp3_size:.2f} MB")

            if keep_files:
                print(f"\n📁 Intermediate files kept:")
                if os.path.exists(md_path):
                    print(f"  • Markdown: {md_path}")
                if os.path.exists(final_wav_path):
                    print(f"  • WAV audio: {final_wav_path}")

        except Exception as e:
            print(f"Error during full pipeline conversion: {e}", file=sys.stderr)

            # Cleanup on error
            if 'md_path' in locals() and not keep_files:
                try:
                    if os.path.exists(md_path):
                        os.unlink(md_path)
                except Exception:
                    pass

            if 'final_wav_path' in locals() and not keep_files:
                try:
                    if os.path.exists(final_wav_path):
                        os.unlink(final_wav_path)
                except Exception:
                    pass

            sys.exit(1)


if __name__ == "__main__":
    main()
