from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile as wav

class Speaker:
    def __init__(self, tts_model: str = None):
        if not tts_model or tts_model not in Speaker.list_models():            
            tts_model = Speaker.list_models()[0]
            print("Choosing default model: " + tts_model)
                
        self.processor = AutoProcessor.from_pretrained(tts_model)
        self.model = BarkModel.from_pretrained(tts_model).to("mps")
        self.sample_rate = self.model.generation_config.sample_rate

    def list_voices(self):
        return self.processor.model.config.voice_presets

    def text_to_speech(self, text: str, output_file: str, voice_preset: str = None):
        if not voice_preset:
            voice_preset = "v2/en_speaker_6"

        if not output_file:
            output_file = "output.wav"

        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")        
        audio_array = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.processor.tokenizer.pad_token_id
        )
        audio_array = audio_array.cpu().numpy().squeeze()

        with open(output_file, "wb") as f:
            wav.write(f, self.sample_rate, audio_array)

    @staticmethod
    def list_models():
        return [
            "suno/bark"
        ]