#!/usr/bin/env python3
"""
WAV to MP3 converter using pydub.
Can be used as a library or as a standalone script.
"""

import argparse
import sys
import os
import glob
from typing import Optional
from pydub import AudioSegment


def convert_wav_to_mp3(
    wav_path: str,
    output_path: Optional[str] = None,
    bitrate: str = '192k',
    verbose: bool = True
) -> str:
    """
    Convert a WAV file to MP3 format.

    Args:
        wav_path (str): Path to the input WAV file
        output_path (str, optional): Path for the output MP3 file
        bitrate (str): MP3 bitrate (default: '192k')
        verbose (bool): Print progress messages

    Returns:
        str: Path to the generated MP3 file

    Raises:
        FileNotFoundError: If WAV file doesn't exist
        Exception: For conversion errors
    """
    # Validate input file
    if not os.path.exists(wav_path):
        # Provide helpful error message with available WAV files
        error_msg = f"WAV file not found: {wav_path}"
        if verbose:
            print(f"Error: {error_msg}", file=sys.stderr)
            print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
            wav_files = glob.glob("*.wav")
            if wav_files:
                print("Available WAV files:", file=sys.stderr)
                for wav_file in wav_files:
                    print(f"  - {wav_file}", file=sys.stderr)
            else:
                print("No WAV files found in current directory", file=sys.stderr)
        raise FileNotFoundError(error_msg)

    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(wav_path)[0] + '.mp3'

    # Ensure output path has .mp3 extension
    if not output_path.lower().endswith('.mp3'):
        output_path += '.mp3'

    try:
        # Load the WAV file
        if verbose:
            print(f"Loading WAV file: {wav_path}")

        audio = AudioSegment.from_wav(wav_path)

        # Get audio information
        duration_seconds = len(audio) / 1000.0
        channels = audio.channels
        frame_rate = audio.frame_rate

        if verbose:
            print(f"  Duration: {duration_seconds:.2f} seconds")
            print(f"  Channels: {channels}")
            print(f"  Sample rate: {frame_rate} Hz")

        # Export as MP3
        if verbose:
            print(f"Converting to MP3: {output_path}")
            print(f"  Bitrate: {bitrate}")

        audio.export(output_path, format='mp3', bitrate=bitrate)

        if verbose:
            print(f"✓ Conversion successful!")

        # Print file size comparison if verbose
        if verbose and os.path.exists(output_path):
            wav_size = os.path.getsize(wav_path) / (1024 * 1024)  # MB
            mp3_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            compression_ratio = wav_size / mp3_size if mp3_size > 0 else 0

            print(f"  WAV file size: {wav_size:.2f} MB")
            print(f"  MP3 file size: {mp3_size:.2f} MB")
            if compression_ratio > 0:
                print(f"  Compression ratio: {compression_ratio:.1f}:1")

        return output_path

    except Exception as e:
        error_msg = f"Error during WAV to MP3 conversion: {str(e)}"

        if verbose:
            print(f"Error: {error_msg}", file=sys.stderr)

            # Provide helpful troubleshooting info
            if "ffmpeg" in str(e).lower() or "codec" in str(e).lower():
                print("\nTroubleshooting:", file=sys.stderr)
                print("• You may need to install FFmpeg for MP3 conversion", file=sys.stderr)
                print("• Install FFmpeg from: https://ffmpeg.org/download.html", file=sys.stderr)
                print("• Or use the alternative bitrates: 128k, 256k, 320k", file=sys.stderr)

        raise Exception(error_msg)


def convert_wav_to_mp3_alternative(
    wav_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Alternative conversion method using soundfile (no FFmpeg required).
    Creates a compressed audio format but may not be true MP3.

    Args:
        wav_path (str): Path to the input WAV file
        output_path (str, optional): Path for the output file
        verbose (bool): Print progress messages

    Returns:
        str: Path to the generated file

    Raises:
        ImportError: If required libraries are not available
        Exception: For conversion errors
    """
    try:
        import soundfile as sf
        import numpy as np

        if output_path is None:
            output_path = os.path.splitext(wav_path)[0] + '_compressed.wav'

        if verbose:
            print(f"Using alternative conversion method...")
            print(f"Loading WAV file: {wav_path}")

        # Read WAV file
        data, samplerate = sf.read(wav_path)

        # Write in a compressed format (still WAV but compressed)
        if verbose:
            print(f"Saving compressed audio: {output_path}")

        sf.write(output_path, data, samplerate, subtype='PCM_16')

        if verbose:
            print(f"✓ Alternative conversion complete!")

        return output_path

    except ImportError:
        error_msg = "Alternative method requires soundfile. Install with: pip install soundfile"
        if verbose:
            print(f"Error: {error_msg}", file=sys.stderr)
        raise ImportError(error_msg)

    except Exception as e:
        error_msg = f"Error during alternative conversion: {str(e)}"
        if verbose:
            print(f"Error: {error_msg}", file=sys.stderr)
        raise Exception(error_msg)


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description="Convert WAV file to MP3 format using pydub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Bitrate options:
  128k  - Good quality, smaller file size
  192k  - High quality (default)
  256k  - Very high quality
  320k  - Maximum quality, larger file size

Examples:
  python convert_wav2mp3.py audio.wav
  python convert_wav2mp3.py audio.wav --output music.mp3
  python convert_wav2mp3.py audio.wav --bitrate 320k
  python convert_wav2mp3.py audio.wav --alternative  # If FFmpeg issues

Requirements:
  - pydub: pip install pydub
  - FFmpeg: https://ffmpeg.org/download.html (for MP3 support)
        """
    )

    parser.add_argument(
        'wav_file',
        help='Path to the input WAV file'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output MP3 file path (default: same name as WAV with .mp3 extension)'
    )

    parser.add_argument(
        '--bitrate', '-b',
        default='192k',
        help='MP3 bitrate (default: 192k). Options: 128k, 192k, 256k, 320k'
    )

    parser.add_argument(
        '--alternative',
        action='store_true',
        help='Use alternative conversion method (no FFmpeg required, but not true MP3)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.wav_file):
        print(f"Error: WAV file '{args.wav_file}' not found!", file=sys.stderr)
        sys.exit(1)

    if not args.wav_file.lower().endswith('.wav'):
        print(f"Warning: '{args.wav_file}' doesn't have .wav extension", file=sys.stderr)

    verbose = not args.quiet

    try:
        if args.alternative:
            # Use alternative conversion method
            output_path = convert_wav_to_mp3_alternative(
                wav_path=args.wav_file,
                output_path=args.output,
                verbose=verbose
            )
        else:
            # Use standard pydub conversion
            output_path = convert_wav_to_mp3(
                wav_path=args.wav_file,
                output_path=args.output,
                bitrate=args.bitrate,
                verbose=verbose
            )

        if verbose:
            print(f"\n✅ Conversion complete!")
            print(f"  Output saved to: {output_path}")

    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)

        # Suggest alternative method if main method failed
        if not args.alternative and "ffmpeg" in str(e).lower():
            print("\nTry using the --alternative flag for a fallback method", file=sys.stderr)

        sys.exit(1)


if __name__ == "__main__":
    main()
