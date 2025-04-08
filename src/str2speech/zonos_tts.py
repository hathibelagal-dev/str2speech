from .base_tts import BaseTTS
import torch
import scipy.io.wavfile as wav
from .zonos.model import Zonos
from .zonos.conditioning import make_cond_dict

class ZonosTTS(BaseTTS):
    model_name = "zyphra/zonos-v0.1-transformer"

    def __init__(self):
        super().__init__()
        self.model = Zonos.from_pretrained(self.model_name, device=self.device)
        self.sample_rate = getattr(self.model.autoencoder, "sampling_rate", 44100)

    def generate(self, prompt, output_file):        
        cond_dict = make_cond_dict(text=prompt, language="en-us")
        conditioning = self.model.prepare_conditioning(cond_dict)
        with torch.no_grad():
            codes = self.model.generate(conditioning)
            audio_array = self.model.autoencoder.decode(codes).cpu()[0]

        audio_array = audio_array.cpu().numpy().squeeze()
        with open(output_file, "wb") as f:
            wav.write(f, self.sample_rate, audio_array)
            print("Audio saved.")
