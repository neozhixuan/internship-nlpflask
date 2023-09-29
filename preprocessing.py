import sys
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
from utils import load_files, tokenize_and_process

N_COMPONENTS = 100  # You can adjust this value as per your requirements for SVD


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Enter your input like this: python preprocessing.py corpus")

    files = load_files(sys.argv[1])

    file_words = {
        filename: tokenize_and_process(files[filename])
        for filename in files
    }
    # corpus_directory = sys.argv[1]
    # files = load_files(corpus_directory)

    # for filename in files:
    #     file = files[filename]
    #     if (is_pdf(file)):
    #         file_path = os.path.join(corpus_directory, filename)
    #         pdf_text = load_pdf_text(file_path)
    #         processed_words = tokenize_and_process(pdf_text)
    #         file = processed_words

    #     file_words = {
    #         filename: tokenize_and_process(file)
    #     }
    # Initialise libraries
    vectorizer = TfidfVectorizer()
    # Fit the SVD model
    svd = TruncatedSVD(n_components=N_COMPONENTS)
    # You MUST fit the vectorizer with training data to learn vocab + IDF values
    # in order to transform the query vector
    tfidf_matrix = vectorizer.fit_transform(
        " ".join(words) for words in file_words.values())

    # Save the TF-IDF vectorizer and matrix to files
    with open("preprocess/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("preprocess/tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    latent_semantic_matrix = svd.fit_transform(tfidf_matrix)

    # Save the SVD model and latent semantic matrix to files
    with open("preprocess/svd_model.pkl", "wb") as f:
        pickle.dump(svd, f)

    with open("preprocess/latent_semantic_matrix.pkl", "wb") as f:
        pickle.dump(latent_semantic_matrix, f)

    print("Successfully preprocessed all vectors and matrices.")


if __name__ == "__main__":
    main()
