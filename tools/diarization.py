import json
import os
import subprocess
import argparse
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from collections import defaultdict


def process_audio(audio_path, output_dir):
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set. "
            "Get a token at https://huggingface.co/settings/tokens"
        )

    subprocess.run([
        "whisperx", str(audio_path),
        "--language", "en",
        "--diarize",
        "--hf_token", hf_token,
        "--output_dir", str(output_dir),
        "--output_format", "json",
        "--min_speakers", "2",
    ], check=True)

    with open(output_dir / f"{audio_path.stem}.json", "r", encoding="utf-8") as f:
        segments = json.load(f).get("segments", [])

    audio, sr = librosa.load(audio_path, sr=16000)
    speaker_audio = defaultdict(list)
    speaker_text = defaultdict(str)

    for seg in segments:
        if "speaker" in seg:
            start, end = int(seg["start"] * sr), int(seg["end"] * sr)
            speaker_audio[seg["speaker"]].append(audio[start:end])
            if "text" in seg:
                speaker_text[seg["speaker"]] += seg["text"] + " "

    for speaker, chunks in speaker_audio.items():
        out_path = output_dir / f"{audio_path.stem}_{speaker}.wav"
        sf.write(out_path, np.concatenate(chunks), sr)
        print(f"Saved: {out_path}")

        text_path = output_dir / f"{audio_path.stem}_{speaker}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(speaker_text[speaker].strip())
        print(f"Saved: {text_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()
    process_audio(args.audio_file, Path(args.output) if args.output else Path(args.audio_file).parent)
