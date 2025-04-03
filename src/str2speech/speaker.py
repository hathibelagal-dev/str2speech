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
            from .bark_tts import BarkTTS
            self.model = BarkTTS(tts_model)
        elif "mms-tts" in tts_model:
            from .mms_tts import MMSTTS
            self.model = MMSTTS(tts_model)
        elif "zonos" in tts_model:
            from .zonos_tts import ZonosTTS
            self.model = ZonosTTS()
        elif "kokoro" in tts_model:
            from .kokoro_tts import KokoroTTS
            self.model = KokoroTTS()
        elif "sesame" in tts_model:
            from .sesame_tts import SesameTTS
            self.model = SesameTTS()
        elif "spark" in tts_model:
            from .spark_tts import SparkTTS
            self.model = SparkTTS()
        elif "speecht5" in tts_model:
            from .speecht5_tts import SpeechT5TTS
            self.model = SpeechT5TTS()
        elif "megatts3" in tts_model:
            from .mega3_tts import Mega3TTS
            self.model = Mega3TTS()

    def text_to_speech(self, text: str, output_file: str, voice_preset: str = None, speed: float = 1.0):
        if "bark" in self.tts_model or "kokoro" in self.tts_model:
            if voice_preset:
                self.model.voice_preset = voice_preset
                if "kokoro" in self.tts_model:
                    self.model.change_voice()
            self.model.speed = float(speed)
            self.model.generate(text, output_file)
        elif (
            "mms-tts" in self.tts_model
            or "zonos" in self.tts_model
            or "sesame" in self.tts_model
            or "spark" in self.tts_model
            or "speecht5" in self.tts_model
            or "megatts3" in self.tts_model
        ):
            if voice_preset:
                print(
                    "WARNING: Voice presets are currently not supported for this model."
                )
            self.model.generate(text, output_file)

    @staticmethod
    def list_models():
        return [
            {"name": "suno/bark-small"},
            {"name": "suno/bark"},
            {"name": "megatts3"},
            {"name": "facebook/mms-tts-eng"},
            {"name": "facebook/mms-tts-deu"},
            {"name": "facebook/mms-tts-fra"},
            {"name": "facebook/mms-tts-spa"},
            {"name": "facebook/mms-tts-swe"},
            {"name": "kokoro"},
            {"name": "sesame/csm-1b"},
            {"name": "zyphra/zonos-v0.1-transformer"},
            {"name": "sparkaudio/spark-tts-0.5b"},
            {"name": "microsoft/speecht5_tts"}
        ]
