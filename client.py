import argparse
import os
from RealtimeSTT import AudioToTextRecorder
from core.audio_player_manager import AudioPlayerManager
from core.tts_client import TTSClient


def process_text(text, voice_client: TTSClient, player: AudioPlayerManager):
    print(f"\nYou said: '{text}'")
    player.play_stream(voice_client.generate_voice_streaming(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time voice swap client")
    parser.add_argument("--ref-audio", required=True, help="Path to reference audio file")
    parser.add_argument("--ref-text", required=True, help="Transcript of the reference audio")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    args = parser.parse_args()

    voice_client = TTSClient(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        server_url=args.server,
    )
    recorder = AudioToTextRecorder(language="en")
    player = AudioPlayerManager()

    try:
        while True:
            text = recorder.text()
            process_text(text, voice_client, player)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        player.stop()
        recorder.shutdown()
        os._exit(0)
