from .base_tts import BaseTTS
from .megatts3.tts.infer_cli import generate as G
from .utils import get_downloads_path
import requests
import os
from huggingface_hub import snapshot_download

class Mega3TTS(BaseTTS):
    def __init__(self):
        super().__init__()
        self.model_name = "megatts3"
        default_voice_asset1 = "https://github.com/hathibelagal-dev/str2speech/raw/refs/heads/main/assets/voices/megatts_p.wav"
        default_voice_asset2 = "https://github.com/hathibelagal-dev/str2speech/raw/refs/heads/main/assets/voices/megatts_p.npy"
        self.model_dir = get_downloads_path("megatts3")
        if not os.path.exists(self.model_dir):
            snapshot_download(repo_id="ByteDance/MegaTTS3", local_dir=self.model_dir)
            os.makedirs(os.path.join(self.model_dir, "voices"))
            response = requests.get(default_voice_asset1)
            if response.status_code == 200:
                with open(
                    os.path.join(self.model_dir, "voices", "megatts_p.wav"), "wb"
                ) as f:
                    f.write(response.content)
                    print("Default voice asset 1 downloaded")
            response = requests.get(default_voice_asset2)
            if response.status_code == 200:
                with open(
                    os.path.join(self.model_dir, "voices", "megatts_p.npy"), "wb"
                ) as f:
                    f.write(response.content)
                    print("Default voice asset 2 downloaded")        

    def generate(self, prompt, output_file):
        G(
            prompt,
            os.path.join(self.model_dir, "voices", "megatts_p.wav"),
            output_file=output_file
        )