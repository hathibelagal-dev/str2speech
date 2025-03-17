import os
import sys

def is_colab():
    return 'COLAB_GPU' in os.environ or 'google.colab' in sys.modules