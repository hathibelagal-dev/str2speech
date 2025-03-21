from .base_tts import BaseTTS
from transformers import VitsTokenizer, VitsModel, AutoProcessor
import scipy.io.wavfile as wav
import torch


class MMSTTS(BaseTTS):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = VitsModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = VitsTokenizer.from_pretrained(model_name)
        self.sample_rate = 16000

    def generate(self, prompt, output_file):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            audio_array = outputs.waveform[0]

        audio_array = audio_array.cpu().numpy().squeeze()
        with open(output_file, "wb") as f:
            wav.write(f, self.sample_rate, audio_array)
            print("Audio saved.")
