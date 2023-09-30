
import fitz  # PyMuPDF
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')


def load_files(directory):
    """
    Given a directory name containing docx, return a dictionary mapping the filename of each
    `.docx` file inside that directory to the file's contents as a string.
    """
    dic = {}
    for filename in os.listdir(directory):
        newname = os.path.join(directory, filename)
        if (is_pdf(newname)):
            pdf_text = load_pdf_text(newname)
            dic[filename] = pdf_text
        else:
            with open(newname, encoding='utf8') as file:
                dic[filename] = file.read()
    return dic


def tokenize_and_process(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Preprocess document by
    1. Converting all words to lowercase
    2. Removing any punctuation or English stopwords.

    Then, it lemmatises each word to reduce feature space.
    """
    stemmer = PorterStemmer()
    # Lowercase all words
    text = document.lower()
    tokens = word_tokenize(text)
    stopw = stopwords.words('english')
    newtokens = []
    # Exclude stopwords and punctuations
    for token in tokens:
        if (token not in stopw) and (token not in punctuation):
            # Lemmatise the words (cuts down on about 40% of entries)
            newtokens.append(stemmer.stem(token))
    return newtokens


def load_pdf_text(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    text = ""
    # Iterate through pages and extract text
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()

    return text


def is_pdf(file_path):
    _, extension = os.path.splitext(file_path)
    return extension.lower() == ".pdf"


def load_document(directory, filename):
    newname = os.path.join(directory, filename)

    # Extract text from pdf
    pdf_text = ""
    if (is_pdf(newname)):
        pdf_text = load_pdf_text(newname)
    else:
        with open(newname, encoding='utf8') as file:
            pdf_text = file.read()
    return pdf_text
