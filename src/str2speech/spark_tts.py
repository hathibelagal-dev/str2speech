from huggingface_hub import snapshot_download
from .utils import get_downloads_path
import os
from .base_tts import BaseTTS

class SparkTTS(BaseTTS):
    model_name = "SparkAudio/Spark-TTS-0.5B"
    def __init__(self):
        super().__init__()
        self.download_model()

    def download_model(self):
        self.model_dir = get_downloads_path("SparkTTS")
        if not os.path.exists(self.model_dir):
            snapshot_download(self.model_name, local_dir=self.model_dir)
        else:
            print("Model already downloaded")