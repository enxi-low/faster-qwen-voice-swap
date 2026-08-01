# FasterQwen Voice Swap

Real-time voice swap using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) voice cloning. Speak into your mic and your words play back in a cloned target voice.

```
Microphone → Whisper STT → Qwen3-TTS (voice clone) → Speaker
```

A local FastAPI server runs the Qwen3-TTS model and streams audio chunks back. A client captures your mic, transcribes speech in real time, and feeds the text to the server.

> **note:** This is an STT-TTS pipeline. It waits for a complete sentence before generating audio. Expect lag per sentence depending on your GPU and sentence length.

## Prerequisites

- Python 3.12 installed before running the installer
- NVIDIA CUDA GPU for the 3 Swap Cam integration
- [ffmpeg](https://ffmpeg.org/) on PATH (for voice preparation tools)
- A Hugging Face account/token with access to [pyannote speaker diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) when using the voice preparation tools

## Installation

Use the platform installer:

- **Windows:** run `install.bat`
- **Linux/macOS:** run `./install.sh`

The installer creates two separate environments:

- `.venv` for the FasterQwen3TTS server and RealtimeSTT runtime
- `.venv_tools` for WhisperX/pyannote voice preparation

It also downloads DeepFilterNet 0.5.6, installs FFmpeg if needed, and asks for the Hugging Face token used by speaker diarization.

For a manual runtime-only installation:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you also want to use the voice preparation tools:

```bash
python3.12 -m venv .venv_tools
source .venv_tools/bin/activate
pip install -r requirements-tools.txt
python install.py
```

On Windows, use `.venv\Scripts\activate` and `.venv_tools\Scripts\activate` instead.

Download the [DeepFilter binary](https://github.com/Rikorose/DeepFilterNet/releases) for your platform, rename it to `deep-filter.exe` and place it in the project root.

Set your HuggingFace token (needed for speaker diarization) and accept the licence at: https://huggingface.co/pyannote/speaker-diarization-3.1. Run:  
```bash
cp .env.example .env
```
Edit .env and set HF_TOKEN=your_token

> **Tip:** Keep the runtime and preparation dependencies in their separate virtual environments because `faster-qwen3-tts` and `whisperx` can have conflicting torch/CUDA dependencies.

## Quick start

### 1. Get a reference voice sample

You need a clean 10â€“30s audio clip of the target voice and its transcript.

**Already have a clean clip?** Skip to step 2.

**Starting from a raw mixed recording?** Use the preparation tools:

```bash
source .venv_tools/bin/activate
python prepare_voice.py recording.mp3 --output output_dir
```

This produces a `.wav` and `.txt` per detected speaker. See [`examples/diarization/`](examples/diarization/) for a real before/after â€” `input/mixed_with_music.mp3` is a raw two-speaker recording, `output/` contains the cleaned per-speaker files.

### 2. Start the server

```bash
source .venv/bin/activate
python server.py --port 8000
```

The first run downloads the Qwen3-TTS model (~2 GB). On Windows you can use `start_server.bat`.

### 3. Start the client

```bash
python client.py --ref-audio path/to/voice.wav --ref-text "The transcript of that clip."

e.g

python client.py --ref-audio "examples/diarization/output/speaker_00.wav" --ref-text "This year, in the year of the horse, I hope we continue to do well. That's great to hear, Mr. Lee. This year, in the year of the horse, I hope we continue to do well."
```

Speak into your mic. Your words will play back in the cloned voice.

## Project structure

```
server.py                     FastAPI server wrapping Qwen3-TTS
client.py                     Mic â†’ STT â†’ server â†’ speaker
prepare_voice.py              CLI: denoise + diarize a recording
install.py                    DeepFilter, FFmpeg, and Hugging Face token setup
requirements.txt              Runtime environment
requirements-tools.txt        Voice-preparation environment
core/
  tts_client.py               HTTP client for the server API
  audio_player_manager.py     Sounddevice output stream
tools/
  cleaning.py                 DeepFilterNet noise removal
  diarization.py              WhisperX speaker diarization
```
