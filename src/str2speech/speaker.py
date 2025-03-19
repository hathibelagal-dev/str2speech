from .kokoro_tts import KokoroTTS
from .bark_tts import BarkTTS
from .sesame_tts import SesameTTS
from .mms_tts import MMSTTS
from .zonos_tts import ZonosTTS
from .spark_tts import SparkTTS

class Speaker:
    def __init__(self, tts_model: str = None):
        if tts_model:
            tts_model = tts_model.lower()
        print(f"Model provided: {tts_model}")
        import logging as l        
        l.getLogger("torch").setLevel(l.ERROR)        
        if not tts_model or tts_model not in [
            model["name"] for model in Speaker.list_models()
        ]:
            tts_model = Speaker.list_models()[0]["name"]
            print("Choosing default model: " + tts_model)

        self.tts_model = tts_model
        if "bark" in tts_model:
            self.model = BarkTTS(tts_model)
        elif "mms-tts" in tts_model:            
            self.model = MMSTTS(tts_model)
        elif "zonos" in tts_model:
            self.model = ZonosTTS()
        elif "kokoro" in tts_model:
            self.model = KokoroTTS()            
        elif "sesame" in tts_model:
            self.model = SesameTTS()
        elif "spark" in tts_model:            
            self.model = SparkTTS()           

    def text_to_speech(self, text: str, output_file: str, voice_preset: str = None):
        if "bark" in self.tts_model or "kokoro" in self.tts_model:
            if voice_preset:
                self.model.voice_preset = voice_preset
            self.model.generate(text, output_file)
        elif "mms-tts" in self.tts_model or "zonos" in self.tts_model or "sesame" in self.tts_model or "spark" in self.tts_model:
            if voice_preset:
                print("WARNING: Voice presets are currently not supported for this model.")
            self.model.generate(text, output_file)

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
            {"name": "sparkaudio/spark-tts-0.5b"}
        ]
