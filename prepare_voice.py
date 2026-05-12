"""
Prepare a voice reference sample from a mixed recording.

Steps:
  1. Noise removal using DeepFilterNet
  2. Speaker diarization using WhisperX + pyannote

Outputs per-speaker WAV files and transcripts in the output directory.
Use the resulting .wav + .txt as --ref-audio / --ref-text for the client.
"""
import argparse
from pathlib import Path
from tools.diarization import process_audio
from tools.cleaning import AudioCleaner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and diarize audio to extract per-speaker voice samples."
    )
    parser.add_argument("audio_file", help="Path to input audio file")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path.cwd() / f"output_{Path(args.audio_file).stem}"
    output_dir.mkdir(exist_ok=True, parents=True)

    cleaner = AudioCleaner()
    cleaned_path = cleaner.clean_audio(args.audio_file, output_dir)
    print(f"Cleaned audio: {cleaned_path}")

    process_audio(cleaned_path, output_dir)
