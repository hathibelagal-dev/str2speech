from kokoro import KPipeline
import soundfile as sf

class KokoroTTS:
    def __init__(self, voice_preset:str = "af_heart"):
        self.pipeline = KPipeline(lang_code='a', repo_id="hexgrad/Kokoro-82M")
        self.voice_preset = voice_preset
        self.sample_rate = 24000

    def generate(self, prompt, output):
        g = self.pipeline(
            prompt, voice=self.voice_preset,
            speed=1
        )
        for _, (_, _, audio) in enumerate(g):
            sf.write(output, audio, self.sample_rate)
        print("Audio saved.")

