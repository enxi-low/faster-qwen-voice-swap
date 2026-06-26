import argparse
import asyncio
import os
import base64
import json
import queue
import shutil
import tempfile
import threading

from RealtimeSTT.audio_recorder import AudioToTextRecorder
from fastapi import FastAPI, File, HTTPException, Response, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import uvicorn

try:
    from faster_qwen3_tts import FasterQwen3TTS
except ImportError:
    print("Warning: faster_qwen3_tts not found. Install it to run the server.")
    FasterQwen3TTS = None

app = FastAPI(title="FasterQwen3TTS Server")

model = None
warmup_ref_audio = None
warmup_ref_text = None

recorder = None
response_queue = queue.Queue()

language: str = "English"
ref_audio: str = "audio.wav"
ref_text: str = ""
ref_text_path: str = "text.txt"
chunk_size: int = 8

stt_task = None


class VoiceCloneRequest(BaseModel):
    text: str
    language: str = "English"
    ref_audio: str = ref_audio
    ref_text: str = ref_text
    chunk_size: int = 8


@app.on_event("startup")
async def load_model():
    global model
    print("Loading FasterQwen3TTS model...")
    model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    print("Model loaded.")

    if warmup_ref_audio and warmup_ref_text:
        print("Running warm-up inference (CUDA graph capture)...")
        def run_warmup():
            for _ in model.generate_voice_clone_streaming(
                text="Hello.",
                language="English",
                ref_audio=warmup_ref_audio,
                ref_text=warmup_ref_text,
                chunk_size=8,
            ):
                pass
        await asyncio.to_thread(run_warmup)
        print("Warm-up complete. Server ready.")
    else:
        print("No --ref-audio provided — first request will trigger CUDA warm-up and be slower.")
    
        if os.path.exists(ref_text_path):
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text = f.read()

    global recorder
    recorder = AudioToTextRecorder(language="en", use_microphone=False, post_speech_silence_duration=0.4)
    global stt_task
    stt_task = asyncio.create_task(stt_worker())


@app.on_event("shutdown")
async def shutdown():
    if stt_task:
        stt_task.cancel()
    if recorder:
        recorder.stop()


async def stt_worker():
    while True:
        text = await asyncio.to_thread(recorder.text)
        if text:
            print("STT transcription received: %s", text)
            response_queue.put(
                await generate_voice(
                    VoiceCloneRequest(
                        text=text,
                        language=language,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        chunk_size=chunk_size,
                    )
                )
            )


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/feed_audio")
async def feed_audio(request: Request):
    pcm_bytes = await request.body()
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    recorder.feed_audio(pcm)

    try:
        resp = response_queue.get_nowait()
        return resp
    except queue.Empty:
        return Response(status_code=204)


@app.post("/generate_voice")
async def generate_voice(request: VoiceCloneRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    print(f"Received: '{request.text}'")

    q: queue.Queue = queue.Queue()

    def run_inference():
        try:
            for audio_chunk, sample_rate, _ in model.generate_voice_clone_streaming(
                text=request.text,
                language=request.language,
                ref_audio=request.ref_audio,
                ref_text=request.ref_text,
                chunk_size=request.chunk_size,
            ):
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.squeeze()
                line = json.dumps({
                    "audio_b64": base64.b64encode(audio_chunk.astype(np.float32).tobytes()).decode(),
                    "samplerate": int(sample_rate),
                }) + "\n"
                q.put(line)
        except Exception as e:
            q.put(json.dumps({"error": str(e)}) + "\n")
        finally:
            q.put(None)

    threading.Thread(target=run_inference, daemon=True).start()

    async def generate():
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                break
            yield item

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/set_ref_voice")
async def set_ref_voice(audio: UploadFile = File(...), text: UploadFile = File(...)):
    global ref_text

    os.makedirs(os.path.dirname(ref_audio) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(ref_text_path) or ".", exist_ok=True)
    tmp_audio_path = None
    try:
        # Handle audio
        fd, tmp_audio_path = tempfile.mkstemp(
            suffix=".wav",
            dir=os.path.dirname(ref_audio) or ".",
        )
        with os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        os.replace(tmp_audio_path, ref_audio)

        # Handle text
        content = await text.read()
        ref_text = content.decode("utf-8")
        with open(ref_text_path, "w", encoding="utf-8") as f:
            f.write(ref_text)

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        await audio.close()
        await text.close()
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

    return {"ok": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FasterQwen3TTS server")
    parser.add_argument("--ref-audio", default=None, help="Reference audio for startup warm-up")
    parser.add_argument("--ref-text", default=None, help="Transcript of the warm-up reference audio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    warmup_ref_audio = args.ref_audio
    warmup_ref_text = args.ref_text

    uvicorn.run(app, host=args.host, port=args.port)
