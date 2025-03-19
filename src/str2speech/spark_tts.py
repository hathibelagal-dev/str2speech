from huggingface_hub import snapshot_download

class SparkTTS:
    def __init__(self):
        super().__init__()
        self.model_name = "SparkAudio/Spark-TTS-0.5B"

    def download_model(self):
        pass