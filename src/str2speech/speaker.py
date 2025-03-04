from transformers import AutoProcessor, BarkModel
from transformers import VitsTokenizer, VitsModel

import scipy.io.wavfile as wav
import torch
import sys

class Speaker:
    def __init__(self, tts_model: str = None):
        if not tts_model or tts_model not in [
            model["name"] for model in Speaker.list_models()
        ]:
            tts_model = Speaker.list_models()[0]["name"]
            print("Choosing default model: " + tts_model)

        self.tts_model = tts_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if "Zonos" not in tts_model:
            self.processor = AutoProcessor.from_pretrained(tts_model)

        if "bark" in tts_model:
            self.model = BarkModel.from_pretrained(tts_model).to(self.device)
            if self.device != "cpu":
                self.model.enable_cpu_offload()
            self.sample_rate = self.model.generation_config.sample_rate
        elif "mms-tts" in tts_model:
            self.model = VitsModel.from_pretrained(tts_model).to(self.device)
            self.tokenizer = VitsTokenizer.from_pretrained(tts_model)
            self.sample_rate = 16000
        elif "zonos" in tts_model.lower():
            try:
                from zonos.model import Zonos                
                self.model = Zonos.from_pretrained(tts_model, device=self.device)
                self.sample_rate = getattr(self.model.autoencoder, "sampling_rate", 44100)
            except ImportError:
                print("Note: Zonos model requires the zonos package.")                
                sys.exit(1)            

    def text_to_speech(self, text: str, output_file: str, voice_preset: str = None):
        if "bark" in self.tts_model:
            if not voice_preset:
                voice_preset = "v2/en_speaker_6"
                print("Using default voice preset: " + voice_preset)
            inputs = self.processor(
                text, voice_preset=voice_preset, return_tensors="pt"
            )
            audio_array = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        elif "mms-tts" in self.tts_model:
            if voice_preset:
                print("WARNING: Voice presets are not supported for this model.")
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                audio_array = outputs.waveform[0]
        elif "zonos" in self.tts_model.lower():
            from zonos.conditioning import make_cond_dict
            cond_dict = make_cond_dict(text=text, language="en-us")
            conditioning = self.model.prepare_conditioning(cond_dict)
            with torch.no_grad():
                codes = self.model.generate(conditioning)
                audio_array = self.model.autoencoder.decode(codes).cpu()[0]

        if audio_array is not None:
            audio_array = audio_array.cpu().numpy().squeeze()
            with open(output_file, "wb") as f:
                wav.write(f, self.sample_rate, audio_array)
        else:
            print("ERROR: Couldn't generate speech.")

    @staticmethod
    def list_models():
        return [
            {"name": "suno/bark-small"},
            {"name": "suno/bark"},
            {"name": "facebook/mms-tts-eng"},
            {"name": "facebook/mms-tts-deu"},
            {"name": "facebook/mms-tts-fra"},
            {"name": "facebook/mms-tts-spa"},
            {"name": "Zyphra/Zonos-v0.1-transformer"}
        ]
