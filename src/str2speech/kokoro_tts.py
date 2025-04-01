import os
from kokoro import KPipeline
import soundfile as sf
from .base_tts import BaseTTS


class KokoroTTS(BaseTTS):
    def __init__(self, voice_preset: str = "af_heart"):
        super().__init__()
        self.pipeline = KPipeline(
            lang_code="a", repo_id="hexgrad/Kokoro-82M", device=self.device
        )
        self.voice_preset = voice_preset
        self.sample_rate = 24000
        self.speed = 1.0        

    def generate(self, prompt, output_file):
        g = self.pipeline(prompt, voice=self.voice_preset, speed=self.speed)
        i = 0

        if "/" in output_file:
            seperator = os.path.sep
            directory = seperator.join(output_file.split(seperator)[:-1])
            output_file = output_file.split(seperator)[-1]
            if not os.path.exists(directory):
                print("Provided directory does not exist: " + directory)
                os.makedirs(directory)
                print("Created directory: " + directory)
        else:
            directory = "./"
        for item in g:
            _output_file = os.path.join(directory, f"{i}_{output_file}")
            sf.write(_output_file, item.output.audio, self.sample_rate)
            i += 1
            print("Audio saved to " + _output_file)
