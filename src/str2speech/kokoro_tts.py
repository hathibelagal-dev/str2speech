from kokoro import KPipeline
import soundfile as sf
from .base_tts import BaseTTS

class KokoroTTS(BaseTTS):
    def __init__(self, voice_preset:str = "af_heart"):
        super().__init__()
        self.pipeline = KPipeline(
            lang_code='a', 
            repo_id="hexgrad/Kokoro-82M", device=self.device
        )
        self.voice_preset = voice_preset
        self.sample_rate = 24000

    def generate(self, prompt, output_file):
        g = self.pipeline(
            prompt, voice=self.voice_preset,
            speed=1
        )
        for item in g:
            sf.write(output_file, item.output.audio, self.sample_rate)
            print("Audio saved.")

