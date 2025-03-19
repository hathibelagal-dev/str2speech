import sys
from .cloner import Cloner
from .base_tts import BaseTTS

class ZonosTTS(BaseTTS):
    model_name = "zyphra/zonos-v0.1-transformer"
    def __init__(self):
        try:
            from zonos.model import Zonos                
            self.model = Zonos.from_pretrained(self.model_name, device=self.device)
            self.sample_rate = getattr(self.model.autoencoder, "sampling_rate", 44100)
        except ImportError:
            print("Note: Zonos model requires the zonos package.")
            Cloner.clone_and_install("https://github.com/hathibelagal-dev/Zonos.git")
            print("Please re-run str2speech for the Zonos model to work.")
            sys.exit(0)