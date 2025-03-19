from huggingface_hub import snapshot_download
from .utils import get_downloads_path
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP

class SparkTTS:
    model_name = "SparkAudio/Spark-TTS-0.5B"
    def __init__(self):
        super().__init__()
        self.download_model()
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.model.to(self.device)

    def download_model(self):
        self.model_dir = get_downloads_path("SparkTTS")
        if not os.path.exists(self.model_dir):
            snapshot_download(self.model_name, local_dir=self.model_dir)
        else:
            print("Model already downloaded")