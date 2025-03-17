import str2speech
import argparse
import time
from transformers import logging
import str2speech.speaker as speaker

import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def main():
    print(f"Now running str2speech {str2speech.__version__}")

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
    args = parser.parse_args()

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
        s.text_to_speech(text, output, args.voice)
        end_time = time.time()
        print(f"Generated speech in {end_time - start_time:.2f} seconds.")
        print(f"Output saved to {output}.")
    except Exception as e:
        print(
            f"ERROR: Couldn't generate speech. Try choosing a different TTS model. Error: {e}"
        )


if __name__ == "__main__":
    main()
