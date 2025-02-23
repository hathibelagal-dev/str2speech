from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile as wav
import torch


class Speaker:
    def __init__(self, tts_model: str = None):
        if not tts_model or tts_model not in Speaker.list_models():
            tts_model = Speaker.list_models()[0]
            print("Choosing default model: " + tts_model["name"])

        self.tts_model = tts_model
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.processor = AutoProcessor.from_pretrained(tts_model["name"])

        if self.tts_model["type"] == 0:
            self.model = BarkModel.from_pretrained(tts_model["name"]).to(self.device)
            self.sample_rate = self.model.generation_config.sample_rate

    def list_voices(self):
        return self.processor.model.config.voice_presets

    def text_to_speech(self, text: str, output_file: str, voice_preset: str = None):
        if not voice_preset:
            if self.tts_model["type"] == 0:
                voice_preset = "v2/en_speaker_6"

        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        audio_array = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        audio_array = audio_array.cpu().numpy().squeeze()

        with open(output_file, "wb") as f:
            wav.write(f, self.sample_rate, audio_array)

    @staticmethod
    def list_models():
        return [
            {"name": "suno/bark-small", "type": 0},
            {"name": "suno/bark", "type": 0},
            {"name": "facebook/mms-tts-eng", "type": 1},
            {"name": "facebook/mms-tts-deu", "type": 1},
            {"name": "facebook/mms-tts-fra", "type": 1},
            {"name": "facebook/mms-tts-spa", "type": 1},
            {"name": "facebook/mms-tts-hin", "type": 1},
        ]
