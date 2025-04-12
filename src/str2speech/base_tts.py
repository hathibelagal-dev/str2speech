import torch
from abc import ABC, abstractmethod


class BaseTTS(ABC):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def generate(self, prompt, output_file):
        pass

    @abstractmethod
    def clone(self, clone_voice):
        pass
