from .sesame_g import load_csm_1b
from huggingface_hub import hf_hub_download
import os
import torchaudio
import torch
import sys

class SesameTTS:
    def __init__(self):                    
        self.model = load_csm_1b("cuda" if torch.cuda.is_available() else "cpu")
        self.voice = None

    def generate(self, prompt, output_file, sample_rate):
        audio = self.model.generate(
            text=prompt,
            speaker=0,
            context=[],
            max_audio_length_ms=120000
        )
        torchaudio.save(output_file, audio.unsqueeze(0).cpu(), sample_rate)