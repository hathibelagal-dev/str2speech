import str2speech
import argparse

import str2speech.speaker as speaker

def main():
    print(f"Now running str2string {str2speech.__version__}")

    parser = argparse.ArgumentParser(description="A tool to convert text to speech.")
    parser.add_argument("--text", "-t", help="The text to convert to speech.", required=True)
    parser.add_argument("--voice", "-v", help="The voice to use. If not provided, the default voice will be used.")
    parser.add_argument("--output", "-o", help="The name of the output file. If not provided, the output will be placed in output.wav.")
    parser.add_argument("--model", "-m", help="The TTS model to use. If not provided, the default model will be chosen.")
    args = parser.parse_args()

    text = args.text
    output = args.output

    try:
        s = speaker.Speaker(args.model)
        s.text_to_speech(text, output)
    except Exception as e:
        print(f"ERROR: Couldn't generate speech. Try choosing a different TTS model. Error: {e}")

if __name__ == "__main__":
    main()
