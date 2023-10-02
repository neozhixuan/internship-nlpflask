import sys
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
from utils import load_files, tokenize_and_process
import time
N_COMPONENTS = 100  # You can adjust this value as per your requirements for SVD


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Enter your input like this: python preprocessing.py corpus")
    start_time = time.time()
    files = load_files(sys.argv[1])

    file_words = {
        filename: tokenize_and_process(files[filename])
        for filename in files
    }

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
    end_time = time.time()
    print(
        f"Successfully preprocessed all vectors and matrices with a time of {(end_time - start_time).to} seconds")


if __name__ == "__main__":
    main()
