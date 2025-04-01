import str2speech
import argparse
import time
from transformers import logging
import str2speech.speaker as speaker
import sys
import os
import warnings

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def main():
    print(f"Now running str2speech {str2speech.__version__}")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description="A tool to convert text to speech.")
    parser.add_argument("--text", "-t", help="The text to convert to speech.")
    parser.add_argument(
        "--file",
        "-f",
        help="The name of the file containing the text to convert to speech.",
    )
    parser.add_argument(
        "--voice",
        "-v",
        help="The voice to use for the Bark model. If not provided, the default voice will be used.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="The name of the output file. If not provided, the output will be placed in output.wav.",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="The TTS model to use. If not provided, the default model will be chosen.",
    )
    parser.add_argument(
        "--speed",
        "-s",
        help="The speed of the speech. If not provided, the default speed will be used. Only supported by Kokoro TTS currently.",
    )
    parser.add_argument(
        "--list-models",
        "-l",
        action="store_true",
        help="List all available TTS models.",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available TTS models:")
        i = 0
        for model in speaker.Speaker.list_models():
            print(f"{i}. {model['name']}")
            i += 1
        sys.exit(0)

    text = args.text
    if not text:
        if args.file:
            with open(args.file, "r") as f:
                text = f.read()
        else:
            text = input("Enter the text you want to convert to speech: ")

    if not text:
        print("ERROR: No text provided.")
        return
    output = args.output if args.output else "output.wav"

    try:
        start_time = time.time()
        s = speaker.Speaker(args.model)
        speed = args.speed
        if(speed):
            speed = float(speed)
            print(f"Using speed: {speed}")
        else:
            speed = 1.0
        s.text_to_speech(text, output, args.voice, speed)
        end_time = time.time()
        print(f"Generated speech in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(
            f"ERROR: Couldn't generate speech. Try choosing a different TTS model. Error: {e}"
        )


if __name__ == "__main__":
    main()
