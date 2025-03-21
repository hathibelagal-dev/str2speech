from .base_tts import BaseTTS
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile as wav


class BarkTTS(BaseTTS):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = BarkModel.from_pretrained(self.model_name).to(self.device)
        self.sample_rate = self.model.generation_config.sample_rate
        if self.device != "cpu" and "small" not in model_name:
            self.model.enable_cpu_offload()
        self.voice_preset = None

    def generate(self, prompt, output_file):
        if not self.voice_preset:
            self.voice_preset = "v2/en_speaker_6"
        inputs = self.processor(
            prompt, voice_preset=self.voice_preset, return_tensors="pt"
        )
        audio_array = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        audio_array = audio_array.cpu().numpy().squeeze()
        with open(output_file, "wb") as f:
            wav.write(f, self.sample_rate, audio_array)
            print("Audio saved.")
