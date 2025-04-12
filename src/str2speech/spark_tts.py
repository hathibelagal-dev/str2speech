from huggingface_hub import snapshot_download
from .utils import get_downloads_path
import os
from .base_tts import BaseTTS
from .cloner import Cloner
import requests
import subprocess


class SparkTTS(BaseTTS):
    model_name = "SparkAudio/Spark-TTS-0.5B"

    def __init__(self):
        super().__init__()
        self.download_model()
        self.voice_preset = None
        self.voice_text = None

    def download_model(self):
        self.model_dir = get_downloads_path("SparkTTS")
        if not os.path.exists(self.model_dir):
            snapshot_download(self.model_name, local_dir=self.model_dir)
            os.makedirs(os.path.join(self.model_dir, "voices"))
            default_voice_asset = "https://github.com/hathibelagal-dev/str2speech/raw/refs/heads/main/assets/voices/generic_female.wav"
            response = requests.get(default_voice_asset)
            if response.status_code == 200:
                with open(
                    os.path.join(self.model_dir, "voices", "generic_female.wav"), "wb"
                ) as f:
                    f.write(response.content)
                    print("Default voice downloaded")
        else:
            print("Model already downloaded")
        try:
            from sparktts.models.audio_tokenizer import BiCodecTokenizer
        except:
            try:
                print("Installing sparktts")
                Cloner.clone_and_install(
                    "https://github.com/hathibelagal-dev/Spark-TTS.git", False
                )
            except:
                print("No installation.")

    def generate(self, prompt, output_file):
        if self.voice_preset is None:
            command = ["sparktts", "--text", prompt, "--save_file", output_file, "--model_dir", self.model_dir]
        else:
            command = [
                "sparktts", "--text", prompt,
                "--save_file", output_file,
                "--model_dir", self.model_dir,
                "--prompt_speech_path", self.voice_preset,
                "--prompt_text", self.voice_text]
        subprocess.run(command)

    def clone(self, clone_voice, voice_text):
        if not voice_text:
            print("Cloning voice failed. No transcript provided. Use the --clone-voice-text argument.")
            return
        if not os.path.exists(clone_voice):
            print("Cloning voice failed. File not found.")
            return
        else:
            self.voice_preset = clone_voice
            self.voice_text = voice_text
            print("Cloning voice...")
