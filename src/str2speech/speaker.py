import scipy.io.wavfile as wav
import torch
from .kokoro_tts import KokoroTTS
from .bark_tts import BarkTTS
from .sesame_tts import SesameTTS
from .mms_tts import MMSTTS
from .zonos_tts import ZonosTTS
import os

class Speaker:
    def __init__(self, tts_model: str = None):
        tts_model = tts_model.lower()
        import logging as l
        l.getLogger("torch").setLevel(l.ERROR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        if not tts_model or tts_model not in [
            model["name"] for model in Speaker.list_models()
        ]:
            tts_model = Speaker.list_models()[0]["name"]
            print("Choosing default model: " + tts_model)

        self.tts_model = tts_model 
        if "bark" in tts_model:
            self.model = BarkTTS("small" if "small" in tts_model else "large")
        elif "mms-tts" in tts_model:            
            self.model = MMSTTS(tts_model)
        elif "zonos" in tts_model:
            self.model = ZonosTTS()
        elif "kokoro" in tts_model:
            self.model = KokoroTTS()            
        elif "sesame" in tts_model:
            self.model = SesameTTS()            

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
        elif "zonos" in self.tts_model:
            from zonos.conditioning import make_cond_dict
            cond_dict = make_cond_dict(text=text, language="en-us")
            conditioning = self.model.prepare_conditioning(cond_dict)
            with torch.no_grad():
                codes = self.model.generate(conditioning)
                audio_array = self.model.autoencoder.decode(codes).cpu()[0]
        elif "kokoro" in self.tts_model or "sesame" in self.tts_model:
            if voice_preset:
                self.model.voice = voice_preset
            self.model.generate(text, output_file, self.sample_rate)
            audio_array = None
            print("Audio saved.")

        if audio_array is not None:
            audio_array = audio_array.cpu().numpy().squeeze()
            with open(output_file, "wb") as f:
                wav.write(f, self.sample_rate, audio_array)
                print("Audio saved.")
        else:
            if "kokoro" not in self.tts_model and "sesame" not in self.tts_model:
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
            {"name": "kokoro"},
            {"name": "sesame/csm-1b"},
            {"name": "zyphra/zonos-v0.1-transformer"},
            {"name": "SparkAudio/Spark-TTS-0.5B"}
        ]
