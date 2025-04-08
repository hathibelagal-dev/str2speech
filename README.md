# 👉 str2speech

![PyPI - Version](https://img.shields.io/pypi/v/str2speech)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/str2speech)
![PyPI - License](https://img.shields.io/pypi/l/str2speech)

## Overview
`str2speech` is a simple command-line tool for converting text to speech using Transformer-based text-to-speech (TTS) models. It supports multiple models and voice presets, allowing users to generate high-quality speech audio from text.

## Latest

We just added support for ByteDance's MegaTTS3. Here's how easy it is to use it:

```
str2speech --model megatts3 --text "This is awesome!"
```

Works fine with just a CPU (needs about 10GB of RAM). But it's always better if you have CUDA available.

---

We now support Microsoft's Speech T5. This is a very lightweight model, and sounds pretty good. Try it out with this:

```bash
str2speech --model "microsoft/speecht5_tts" \
    --text "My dog is prettier than yours." \
    --output "t5test.wav"
```

---

We now support Spark-TTS-0.5B. This is an awesome model. Here's how you use it:

```bash
str2speech --model "SparkAudio/Spark-TTS-0.5B" \
        --text "Hello from Spark" \
        --output "sparktest.wav"
```

---

Added support for Sesame CSM-1B. Here's how to use it:

```bash
export HF_TOKEN=<your huggingface token>
str2speech --text "Hello from Sesame" --model "sesame/csm-1b"
```

---

Added support for Kokoro-82M. This is how you run it:

```bash
str2speech --text "Hello again" --model "kokoro"
```

This is probably the easiest way to use Kokoro TTS.

---

Added support for Zyphra Zonos. Try this out:

```bash
str2speech --text "Hello from Zonos" \
    --model "Zyphra/Zonos-v0.1-transformer" \
    --output hellozonos.wav
```

Alternatively, you could write Python code to use it:

```python
from str2speech.speaker import Speaker

speaker = Speaker("Zyphra/Zonos-v0.1-transformer")
speaker.text_to_speech("Hello, this is a test!", "output.wav")
```

You might need to install `espeak`. Here's how you can install it:

```
sudo apt install espeak-ng
```

## Features
- Supports multiple TTS models, including `Sesame/CSM-1B`, `SparkAudio/Spark-TTS-0.5B`, `Kokoro`, and various `facebook/mms-tts` models.
- Allows selection of voice presets.
- Supports text input via command-line arguments or files.
- Outputs speech in `.wav` format.
- Works with both CPU and GPU.

## Available Models

The following models are supported:
- `Sesame/CSM-1B`
- `MegaTTS3`
- `SparkAudio/Spark-TTS-0.5B`
- `Zyphra/Zonos-v0.1-transformer`
- `microsoft/speecht5_tts`
- `Kokoro` (English and Spanish only)
- `suno/bark-small` (default TTS model)
- `suno/bark`
- `facebook/mms-tts-eng` (English only)
- `facebook/mms-tts-deu` (German only)
- `facebook/mms-tts-fra` (French only)
- `facebook/mms-tts-spa` (Spanish only)
- `facebook/mms-tts-swe` (Swedish only)

## Installation

To install `str2speech`, first make sure you have `pip` installed, then run:

```sh
pip install str2speech
```

## Usage

### Command Line
Run the script via the command line:

```sh
str2speech --text "Hello, world!" --output hello.wav
```

### Options
- `--text` (`-t`): The text to convert to speech.
- `--file` (`-f`): A file containing text to convert to speech.
- `--voice` (`-v`): The voice preset to use (optional, defaults to a predefined voice).
- `--output` (`-o`): The output `.wav` file name (optional, defaults to `output.wav`).
- `--model` (`-m`): The TTS model to use (optional, defaults to `suno/bark-small`).
- `--speed` (`-s`): The speed of the speech (optional, defaults to 1.0). Supported only by Kokoro TTS currently.

Example:
```sh
str2speech --file input.txt --output speech.wav --model suno/bark
```

## API Usage

You can also use `str2speech` as a Python module:

```python
from str2speech.speaker import Speaker

speaker = Speaker()
speaker.text_to_speech("Hello, this is a test.", "test.wav")
```

## Tested With These Dependencies
- `transformers==4.49.0`
- `torch==2.5.1+cu124`
- `numpy==1.26.4`
- `scipy==1.13.1`

## License
This project is licensed under the GNU General Public License v3 (GPLv3).