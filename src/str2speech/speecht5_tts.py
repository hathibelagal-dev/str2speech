from .base_tts import BaseTTS
import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

class SpeechT5TTS(BaseTTS):
    def __init__(self):
        super().__init__()
        self.pipeline = pipeline("text-to-speech", "microsoft/speecht5_tts")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    def generate(self, prompt, output_file):
        speech = self.pipeline(prompt, forward_params={"speaker_embeddings": self.speaker_embedding})
        sf.write(output_file, speech["audio"], speech["sampling_rate"])
        self.sample_rate = speech["sampling_rate"]
        print(f"Audio saved to {output_file}")
        
