from kokoro import KPipeline
import soundfile as sf

class KokoroTTS:
    def __init__(self, voice:str = "af_heart"):
        self.pipeline = KPipeline(lang_code='a')
        self.voice = voice

    def generate(self, prompt, output):
        g = self.pipeline(
            prompt, voice=self.voice,
            speed=1
        )
        for _, (_, _, audio) in enumerate(g):
            sf.write(output, audio, 24000)

