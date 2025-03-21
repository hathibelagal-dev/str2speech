import os
import sys


def is_colab():
    return "COLAB_GPU" in os.environ or "google.colab" in sys.modules


def get_downloads_path(pth: str):
    return os.path.join(os.path.expanduser("~"), ".str2speech", "models", pth)


def get_str2speech_home():
    return os.path.join(os.path.expanduser("~"), ".str2speech")
