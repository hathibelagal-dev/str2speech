from .sesame_g import load_csm_1b
from huggingface_hub import hf_hub_download
import os
import torchaudio

class SesameTTS:
    def __init__(self):
        model_path = os.path.join(os.path.expanduser("~"), '.str2speech', 'sesame', "ckpt.pt")
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            hf_hub_download(repo_id="sesame/csm-1b", filename=model_path, token=os.environ.get("HF_TOKEN"))            
        self.model = load_csm_1b(model_path)
        self.voice = None

    def generate(self, prompt, output_file, sample_rate):
        audio = self.model.generate(
            text=prompt,
            speaker=0,
            context=[],
            max_audio_length_ms=120000
        )
        torchaudio.save(output_file, audio.unsqueeze(0).cpu(), sample_rate)