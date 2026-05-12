import numpy as np
import sounddevice as sd


class AudioPlayerManager:
    def __init__(self, samplerate: int = 24000):
        self.samplerate = samplerate

    def play_stream(self, chunk_iter):
        with sd.OutputStream(samplerate=self.samplerate, channels=1, dtype="float32") as stream:
            for chunk in chunk_iter:
                stream.write(chunk.astype(np.float32))

    def stop(self):
        sd.stop()
