from .base_tts import BaseTTS
from transformers import AutoProcessor, BarkModel

class BarkTTS(BaseTTS):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small").to(self.device)
        self.sample_rate = self.model.generation_config.sample_rate
        if self.device != "cpu":
            self.model.enable_cpu_offload()