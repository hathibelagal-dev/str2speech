from .base_tts import BaseTTS
from .dia.model import Dia
import torch
import soundfile as sf
import numpy as np

class DiaTTS(BaseTTS):
    model_name = "nari-labs/Dia-1.6B"

    def __init__(self):
        super().__init__()
        self.model = Dia.from_pretrained(self.model_name, device=self.device)
        self.voice_preset = None
        self.voice_text = None
        self.sample_rate = 44100

    def generate(self, prompt, output_file):
        output_audio = self.model.generate(
            text=prompt,
            audio_prompt_path=self.voice_preset,
        )
        if isinstance(output_audio, torch.Tensor):
            output_audio = output_audio.cpu().numpy().squeeze()

        if output_audio.ndim == 1:
            output_audio = np.expand_dims(output_audio, axis=0)
        elif (
            output_audio.ndim == 2 and output_audio.shape[0] > output_audio.shape[1]
        ):
            output_audio = output_audio.T
        sf.write(
            output_file, output_audio.squeeze(), self.sample_rate
        )
        print("Audio saved.")

    def clone(self, clone_voice, voice_text = None):
        if clone_voice:
            self.voice_preset = clone_voice
            self.voice_text = voice_text
        else:
            print("Using default voices.")