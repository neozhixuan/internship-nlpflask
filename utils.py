import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation


def load_files(directory):
    """
    Given a directory name containing docx, return a dictionary mapping the filename of each
    `.docx` file inside that directory to the file's contents as a string.
    """
    dic = {}
    for filename in os.listdir(directory):
        newname = os.path.join(directory, filename)
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
