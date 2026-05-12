import argparse
import asyncio
import base64
import json
import queue
import threading

from fastapi import FastAPI, HTTPException
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


class VoiceCloneRequest(BaseModel):
    text: str
    language: str = "English"
    ref_audio: str
    ref_text: str
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


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}


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
