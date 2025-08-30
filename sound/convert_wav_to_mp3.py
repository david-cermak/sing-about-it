from pydub import AudioSegment
import os
import glob

def convert_wav_to_mp3(input_wav_path, output_mp3_path=None, bitrate='192k'):
    """
    Convert a WAV file to MP3 format.

    Args:
        input_wav_path (str): Path to the input WAV file
        output_mp3_path (str): Path for the output MP3 file (optional)
        bitrate (str): MP3 bitrate (default: '192k')
    """
    # If no output path specified, use the same name with .mp3 extension
    if output_mp3_path is None:
        output_mp3_path = input_wav_path.replace('.wav', '.mp3')

    # Check if file exists
    if not os.path.exists(input_wav_path):
        print(f"Error: WAV file '{input_wav_path}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        print("Available WAV files:")
        for wav_file in glob.glob("*.wav"):
            print(f"  - {wav_file}")
        return

    try:
        # Load the WAV file
        print(f"Loading WAV file: {input_wav_path}")
        audio = AudioSegment.from_wav(input_wav_path)

        # Export as MP3
        print(f"Converting to MP3: {output_mp3_path}")
        audio.export(output_mp3_path, format='mp3', bitrate=bitrate)

        print(f"Conversion successful! MP3 saved as: {output_mp3_path}")

        # Print file sizes for comparison
        wav_size = os.path.getsize(input_wav_path) / (1024 * 1024)  # MB
        mp3_size = os.path.getsize(output_mp3_path) / (1024 * 1024)  # MB
        print(f"WAV file size: {wav_size:.2f} MB")
        print(f"MP3 file size: {mp3_size:.2f} MB")
        print(f"Compression ratio: {wav_size/mp3_size:.1f}:1")

    except Exception as e:
        print(f"Error during conversion: {e}")
        print("\nNote: You may need to install FFmpeg for MP3 conversion.")
        print("Install FFmpeg from: https://ffmpeg.org/download.html")
        print("Or try using the alternative conversion method below.")

def convert_wav_to_mp3_alternative(input_wav_path, output_mp3_path=None):
    """
    Alternative conversion method using soundfile and scipy (no FFmpeg required)
    """
    try:
        import soundfile as sf
        import numpy as np
        from scipy.io import wavfile

        if output_mp3_path is None:
            output_mp3_path = input_wav_path.replace('.wav', '.mp3')

        # Read WAV file
        print(f"Loading WAV file: {input_wav_path}")
        data, samplerate = sf.read(input_wav_path)

        # Save as MP3 using scipy (this will create a compressed format)
        print(f"Converting to compressed format: {output_mp3_path}")
        wavfile.write(output_mp3_path, samplerate, data)

        print(f"Conversion successful! File saved as: {output_mp3_path}")

    except ImportError:
        print("scipy not available. Install with: pip install scipy")
    except Exception as e:
        print(f"Error during alternative conversion: {e}")

if __name__ == "__main__":
    # Convert the output.wav file to MP3
    convert_wav_to_mp3('output.wav')

    # If the above fails, try alternative method
    # convert_wav_to_mp3_alternative('output.wav')
