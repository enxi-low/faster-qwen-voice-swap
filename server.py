import argparse
import asyncio
import os
import base64
import json
import queue
import threading

from RealtimeSTT.audio_recorder import AudioToTextRecorder
from fastapi import FastAPI, Form, HTTPException, Response, Request
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
ref_audio: str = ""
ref_text: str = ""
has_warmed_up = False

recorder = None
response_queue = queue.Queue()

language: str = "English"
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
    global model, ref_text, recorder, stt_task
    print("Loading FasterQwen3TTS model...")
    model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    print("Model loaded.")

    if ref_text and ref_audio:
        print("Running warm-up inference (CUDA graph capture)...")
        await asyncio.to_thread(run_warmup)
        print("Warm-up complete. Server ready.")
    else:
        print("No --ref-audio provided — first request will trigger CUDA warm-up and be slower.")

    recorder = AudioToTextRecorder(language="en", use_microphone=False, post_speech_silence_duration=0.4)
    stt_task = asyncio.create_task(stt_worker())


def run_warmup():
    global has_warmed_up
    has_warmed_up = True
    for _ in model.generate_voice_clone_streaming(
        text="Hello.",
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
        chunk_size=8,
    ):
        pass


@app.on_event("shutdown")
async def shutdown():
    if stt_task:
        stt_task.cancel()
        try:
            await stt_task
        except asyncio.CancelledError:
            pass
    if recorder:
        recorder.shutdown()


async def stt_worker():
    try:
        while True:
            text = await asyncio.to_thread(recorder.text)
            if text and has_warmed_up:
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
    except asyncio.CancelledError:
        pass


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
async def set_ref_voice(
    audio_path: str = Form(...),
    text: str = Form(...)
):
    global ref_audio, ref_text
    try:
        if not os.path.exists(audio_path):
            return {"ok": False, "error": f"Audio file not found: {audio_path}"}
        ref_audio = audio_path
        ref_text = text

    except Exception as e:
        return {"ok": False, "error": str(e)}
    
    if not has_warmed_up:
        await asyncio.to_thread(run_warmup)
    
    return {"ok": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FasterQwen3TTS server")
    parser.add_argument("--ref-audio", default=None, help="Reference audio for startup warm-up")
    parser.add_argument("--ref-text", default=None, help="Transcript of the warm-up reference audio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ref_audio = args.ref_audio
    ref_text = args.ref_text

    uvicorn.run(app, host=args.host, port=args.port)
