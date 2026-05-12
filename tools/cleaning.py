import subprocess
from pathlib import Path
import librosa
import soundfile as sf


class AudioCleaner:
    def __init__(self, deepfilter_exe_path=str(Path(__file__).parent.parent / "deep-filter.exe")):
        self.deepfilter_exe_path = deepfilter_exe_path

    def clean_audio(self, input_audio_path, output_dir):
        input_path = Path(input_audio_path)
        output_dir = Path(output_dir)

        audio, sr = librosa.load(str(input_path), sr=None)
        if sr != 48000 or input_path.suffix.lower() != ".wav":
            if sr != 48000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            input_to_clean = output_dir / "cleaned_audio.wav"
            sf.write(str(input_to_clean), audio, 48000)
        else:
            input_to_clean = input_path

        result = subprocess.run(
            [self.deepfilter_exe_path, str(input_to_clean)],
            cwd=str(output_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"DeepFilter failed: {result.stderr}")

        cleaned_path = output_dir / "out" / "cleaned_audio.wav"
        if not cleaned_path.exists():
            raise FileNotFoundError(f"Cleaned audio not found at {cleaned_path}")

        return str(cleaned_path)
