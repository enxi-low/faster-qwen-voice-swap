import sys
import json
import base64
import requests
import numpy as np


SERVER_URL = "http://localhost:8000"


class TTSClient:
    def __init__(self, ref_audio: str, ref_text: str, server_url: str = SERVER_URL):
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.server_url = server_url
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            print(f"Server status: {response.json()}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print("Cannot connect — check the server is running")
            sys.exit(1)

    def _bytes_to_numpy(self, audio_b64: str, dtype: str = "float32") -> np.ndarray:
        return np.frombuffer(base64.b64decode(audio_b64), dtype=dtype)

    def generate_voice_streaming(self, user_text: str, chunk_size: int = 8):
        payload = {
            "text": user_text,
            "language": "English",
            "ref_audio": self.ref_audio,
            "ref_text": self.ref_text,
            "chunk_size": chunk_size,
        }

        print(f"[client] Sending POST for: '{user_text}'")
        try:
            with requests.post(
                f"{self.server_url}/generate_voice",
                json=payload,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "error" in data:
                            print(f"[client] Server error: {data['error']}")
                            return
                        yield self._bytes_to_numpy(data["audio_b64"])
        except Exception as e:
            print(f"[client] Request failed: {type(e).__name__}: {e}")
