from .base_tts import BaseTTS
import torch
import scipy.io.wavfile as wav
from .zonos.model import Zonos
from .zonos.conditioning import make_cond_dict
import torchaudio
import os

class ZonosTTS(BaseTTS):
    model_name = "zyphra/zonos-v0.1-transformer"

    def __init__(self):
        super().__init__()
        self.model = Zonos.from_pretrained(self.model_name, device=self.device)
        self.sample_rate = getattr(self.model.autoencoder, "sampling_rate", 44100)
        self.voice_preset = None
        self.voice_text = None

    def generate(self, prompt, output_file):
        if not self.voice_preset:
            cond_dict = make_cond_dict(text=prompt, language="en-us")
        else:
            _wav, sampling_rate = torchaudio.load(self.voice_preset)
            speaker = self.model.make_speaker_embedding(_wav, sampling_rate)
            cond_dict = make_cond_dict(text=prompt, speaker=speaker, language="en-us")
        conditioning = self.model.prepare_conditioning(cond_dict)
        with torch.no_grad():
            codes = self.model.generate(conditioning)
            if not self.voice_preset:
                audio_array = self.model.autoencoder.decode(codes).cpu()[0]
                audio_array = audio_array.cpu().numpy().squeeze()
                with open(output_file, "wb") as f:
                    wav.write(f, self.sample_rate, audio_array)
                    print("Audio saved.")
            else:
                wavs = self.model.autoencoder.decode(codes).cpu()
                torchaudio.save(output_file, wavs[0], self.model.autoencoder.sampling_rate)
                print("Audio saved.")

    def clone(self, clone_voice, voice_text):
        if not clone_voice:            
            return
        if not os.path.exists(clone_voice):
            print("Cloning voice failed. File not found.")
            return
        else:
            self.voice_preset = clone_voice
            self.voice_text = voice_text
            print("Cloning voice...")
