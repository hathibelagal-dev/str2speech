from . import orpheus
from .base_tts import BaseTTS
import wave

class OrpheusTTS(BaseTTS):
    def __init__(self, model_name: str = "canopylabs/orpheus-3b-0.1-ft"):
        super().__init__()
        self.model_name = model_name
        self.model = orpheus.engine_class.OrpheusModel(model_name=model_name)
        self.voice = "zoe"

    def generate(self, prompt, output_file):
        syn_tokens = self.model.generate_speech(
            prompt=prompt,
            voice=self.voice,
        )

        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)

            total_frames = 0
            chunk_counter = 0
            for audio_chunk in syn_tokens:
                chunk_counter += 1
                frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
                wf.writeframes(audio_chunk)

        print(f"Audio saved to {output_file}.")

