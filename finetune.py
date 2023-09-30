import openai
import fitz
import sys
from utils import load_files, tokenize_and_process

# Set your OpenAI API key
openai.api_key = "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"


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Enter your input like this: python preprocessing.py corpus")

    files = load_files(sys.argv[1])

    # Extract text from pdf
    pdf_text = ""
    for filename in files:
        pdf_text += files[filename]
    print(pdf_text)


if __name__ == "__main__":
    main()
