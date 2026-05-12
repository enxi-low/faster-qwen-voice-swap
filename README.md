# FasterQwen Voice Swap

Real-time voice swap using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) voice cloning. Speak into your mic and your words play back in a cloned target voice.

```
Microphone → Whisper STT → Qwen3-TTS (voice clone) → Speaker
```

A local FastAPI server runs the Qwen3-TTS model and streams audio chunks back. A client captures your mic, transcribes speech in real time, and feeds the text to the server.

> **note:** This is an STT → TTS pipeline, not a signal-level converter like RVC. It waits for a complete sentence before generating audio. Expect ~1–3s lag per sentence depending on your GPU and sentence length.

## Prerequisites

- Python 3.10+
- CUDA GPU strongly recommended (Qwen3-TTS is slow on CPU)
- [ffmpeg](https://ffmpeg.org/) on PATH

## Installation

```bash
pip install -r requirements.txt
```

If you also want to use the voice preparation tools:

```bash
pip install -r requirements-tools.txt
```

> **Tip:** `faster-qwen3-tts` and `whisperx` can have conflicting torch/CUDA dependencies. If you hit issues, install them in separate virtual environments.

## Quick start

### 1. Get a reference voice sample

You need a clean 10–30s audio clip of the target voice and its transcript.

**Already have a clean clip?** Skip to step 2.

**Starting from a raw mixed recording?** Use the preparation tools:

```bash
# Download the DeepFilter binary for your platform:
# https://github.com/Rikorose/DeepFilterNet/releases
# Rename it to deep-filter.exe and place it in the project root.

# Set your HuggingFace token (needed for speaker diarization).
# Also accept the licence at: https://huggingface.co/pyannote/speaker-diarization-3.1
cp .env.example .env
# Edit .env and set HF_TOKEN=your_token

python prepare_voice.py recording.mp3 --output output_dir
```

This produces a `.wav` and `.txt` per detected speaker. See [`examples/diarization/`](examples/diarization/) for a real before/after — `input/mixed_with_music.mp3` is a raw two-speaker recording, `output/` contains the cleaned per-speaker files.

### 2. Start the server

```bash
python server.py
```

The first run downloads the Qwen3-TTS model (~2 GB). On Windows you can use `start_server.bat`.

### 3. Start the client

```bash
python client.py --ref-audio path/to/voice.wav --ref-text "The transcript of that clip."

e.g

python client.py --ref-audio "examples/diarization/output/speaker_00.wav" --ref-text "This year, in the year of the horse, I hope we continue to do well. That's great to hear, Mr. Lee. This year, in the year of the horse, I hope we continue to do well."
```

Speak into your mic. Your words will play back in the cloned voice.

On Windows, edit `start_client.bat` with your paths and run it.

## Project structure

```
server.py                     FastAPI server wrapping Qwen3-TTS
client.py                     Mic → STT → server → speaker
prepare_voice.py              CLI: denoise + diarize a recording
core/
  tts_client.py               HTTP client for the server API
  audio_player_manager.py     Sounddevice output stream
tools/
  cleaning.py                 DeepFilterNet noise removal
  diarization.py              WhisperX speaker diarization
```
