from .base_tts import BaseTTS
import torch
import soundfile as sf
import numpy as np
from .cloner import Cloner


class DiaTTS(BaseTTS):
    model_name = "nari-labs/Dia-1.6B"

    def __init__(self):
        super().__init__()
        self.install_dac_and_dependencies()
        from .dia.model import Dia
        self.model = Dia.from_pretrained(self.model_name, device=self.device, torch_dtype=torch.float16)
        self.voice_preset = None
        self.voice_text = None
        self.sample_rate = 44100

    def install_dac_and_dependencies(self):
        try:
            import audiotools
            print("Audiotools found")
        except:
            print("Installing audiotools")    
            url = "https://github.com/hathibelagal-dev/audiotools.git"
            Cloner.clone_and_install(url, False)
        
        try:
            import dac
            print("Codec found")
        except:
            print("Installing dac")
            url = "https://github.com/hathibelagal-dev/descript-audio-codec.git"
            Cloner.clone_and_install(url, False)

    def generate(self, prompt, output_file):
        output_audio = self.model.generate(
            text=prompt if not self.voice_text else self.voice_text + " " + prompt,
            audio_prompt_path=self.voice_preset,
            cfg_scale=3.0,
            temperature=1.3,
            top_p=0.95,
            use_cfg_filter=True,
            cfg_filter_top_k=30
        )
        if isinstance(output_audio, torch.Tensor):
            output_audio = output_audio.cpu().numpy().squeeze()

        if output_audio.ndim == 1:
            output_audio = np.expand_dims(output_audio, axis=0)
        elif (
            output_audio.ndim == 2 and output_audio.shape[0] > output_audio.shape[1]
        ):
            output_audio = output_audio.T
        sf.write(
            output_file, output_audio.squeeze(), self.sample_rate
        )
        print("Audio saved.")

    def clone(self, clone_voice, voice_text = None):
        if clone_voice:
            self.voice_preset = clone_voice
            self.voice_text = voice_text
        else:
            print("Using default voices.")