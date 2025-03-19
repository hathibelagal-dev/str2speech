# 👉 str2speech 👈 🦾🔊

## Overview
`str2speech` is a simple command-line tool for converting text to speech using Transformer-based text-to-speech (TTS) models. It supports multiple models and voice presets, allowing users to generate high-quality speech audio from text.

## Latest

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
str2speech --text "Hello again" \
    --model "Zyphra/Zonos-v0.1-transformer" \
    --output helloagain.wav
```

Alternatively, you could write Python code to use it:

```python
from str2speech.speaker import Speaker

speaker = Speaker("Zyphra/Zonos-v0.1-transformer")
speaker.text_to_speech("Hello, this is a test!", "output.wav")
```

`str2speech` will try to install Zonos if it doesn't detect it
on your system. You might still have to install `espeak-ng` manually.

If you choose to install Zonos yourself, please run the following:

```bash
apt install espeak-ng
git clone https://github.com/hathibelagal-dev/Zonos.git
cd Zonos && pip install -e .
```

## Features
- Supports multiple TTS models, including `suno/bark-small`, `suno/bark`, and various `facebook/mms-tts` models.
- Allows selection of voice presets.
- Supports text input via command-line arguments or files.
- Outputs speech in `.wav` format.
- Works with both CPU and GPU.

## Available Models

The following models are supported:
- `Sesame/CSM-1B`
- `SparkAudio/Spark-TTS-0.5B`
- `Zyphra/Zonos-v0.1-transformer`
- `Kokoro`
- `suno/bark-small` (default)
- `suno/bark`
- `facebook/mms-tts-eng`
- `facebook/mms-tts-deu`
- `facebook/mms-tts-fra`
- `facebook/mms-tts-spa`

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