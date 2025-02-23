# str2speech

## Overview
`str2speech` is a simple command-line tool for converting text to speech using Transformer-based text-to-speech (TTS) models. It supports multiple models and voice presets, allowing users to generate high-quality speech audio from text.

## Features
- Supports multiple TTS models, including `suno/bark-small`, `suno/bark`, and `facebook/mms-tts-eng`.
- Allows selection of voice presets.
- Outputs speech in `.wav` format.
- Works with both CPU and GPU.

## Installation

To install `str2speech`, first make sure you have `pip` installed, then run:

```sh
pip install str2speech
```

## Usage

### Command Line
Run the script via the command line:

```sh
python -m str2speech.main --text "Hello, world!" --output hello.wav
```

### Options
- `--text` (`-t`): The text to convert to speech (required).
- `--voice` (`-v`): The voice preset to use (optional, defaults to a predefined voice).
- `--output` (`-o`): The output `.wav` file name (optional, defaults to `output.wav`).
- `--model` (`-m`): The TTS model to use (optional, defaults to `suno/bark-small`).

Example:
```sh
python -m str2speech.main -t "This is an AI-generated voice." -o speech.wav -m suno/bark
```

## API Usage

You can also use `str2speech` as a Python module:

```python
from str2speech.speaker import Speaker

speaker = Speaker()
speaker.text_to_speech("Hello, this is a test.", "test.wav")
```

## Available Models

The following models are supported:
- `suno/bark-small` (default)
- `suno/bark`
- `facebook/mms-tts-eng`
- `facebook/mms-tts-deu`
- `facebook/mms-tts-fra`
- `facebook/mms-tts-spa`
- `facebook/mms-tts-hin`

## Dependencies
- `transformers`
- `torch`
- `scipy`

## License
This project is licensed under the GNU General Public License V3.

