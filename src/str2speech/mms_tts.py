from .base_tts import BaseTTS
from transformers import VitsTokenizer, VitsModel, AutoProcessor
    
class MMSTTS(BaseTTS):
    def __init__(self, model_name):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = VitsModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = VitsTokenizer.from_pretrained(model_name)
        self.sample_rate = 16000