from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from utils import load_files, tokenize_and_process
from trainingdata import relevance_labels, documents, cosine_similarity_scores
import joblib
import nltk
import math
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

FILE_MATCHES = 3
SENTENCE_MATCHES = 3


@app.route('/')
def hello():
    return render_template('hello.html')


@app.route('/api/similarity', methods=['POST'])
def similarity_endpoint():
    data = request.get_json()
    query = data.get("query")
    query = query.decode('utf-8')  # Decode input data

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    # Calculate similarity
    similarities = calculate_similarity(query, "corpus")

    # Return the JSON object directly
    response = jsonify({"results": similarities})
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


def calculate_similarity(query, corpus):
    # Check command-line arguments

    # Load the set of files in "corpus"
    files = load_files(corpus)
    file_words = {
        filename: tokenize_and_process(files[filename])
        for filename in files
    }

    list_of_file_names = list(file_words.keys())

    # Load the TF-IDF vectorizer and matrix
    with open("preprocess/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("preprocess/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load the SVD model and latent semantic matrix
    with open("preprocess/svd_model.pkl", "rb") as f:
        svd = pickle.load(f)

    with open("preprocess/latent_semantic_matrix.pkl", "rb") as f:
        latent_semantic_matrix = pickle.load(f)

    # # Prompt user for query
    # original_query = input("Query: ")
    querySet = set(tokenize_and_process(query))
    # print("Query is broken down into", query)
    processed_query = " ".join(querySet)
    ################################################
    # LSI Ranking System (Latenst Semantic Indexing)
    query_vector = vectorizer.transform([processed_query])
    query_latent_vector = svd.transform(query_vector)
    cosine_similarity_lsi = cosine_similarity(
        query_latent_vector, latent_semantic_matrix)
    ################################################
    # TF-IDF Ranking System
    cosine_similarity_tfidf = cosine_similarity(query_vector, tfidf_matrix)
    ################################################
    # Combine the cosine similarities using weights
    combined_cosine_similarity = (
        (0.5*cosine_similarity_lsi) + (0.5*cosine_similarity_tfidf))
    # [[0.02046143 0.08802105 0.39080196 0.60761269 0.12867288 0.18118849]]

    # Sort the indices of the top files, to use it to get file names
    # Eg [3, 2, 5]
    top_file_indices_combined = combined_cosine_similarity.argsort()[
        0][-FILE_MATCHES:][::-1]
    # Eg. Get files 3, 2, 5

    top_files_combined = [list(file_words.keys())[index]
                          for index in top_file_indices_combined]
    relevance_scores_combined = [
        combined_cosine_similarity[0][i] for i in top_file_indices_combined]

    rank_and_relevance = enumerate(
        zip(top_files_combined, relevance_scores_combined), start=1)

    ########################
    # Self Training Algorithm
    # volunteer = (
    #     input("Help to train the machine learning algorithm? (Y/N) "))
    # while not (volunteer.lower() == "y" or volunteer.lower() == "n"):
    #     volunteer = (
    #         input("Help to train the machine learning algorithm? (Y/N)"))

    # if volunteer.lower() == "y":
    #     relevance_ratings = [0] * FILE_MATCHES
    #     count = 0
    #     for document in top_files_combined:
    #         while True:
    #             relevance = input(
    #                 f"How relevant is `{document}` on a scale from 1 to 5: ")
    #             try:
    #                 # Convert the input to float
    #                 relevance = float(relevance)
    #                 if 0 < relevance <= 5:
    #                     relevance_ratings[count] = relevance
    #                     count += 1
    #                     break  # Break out of the while loop if the input is valid
    #                 else:
    #                     print(
    #                         "Invalid input. Relevance score must be between 1 and 5.")
    #             except ValueError:
    #                 print("Invalid input. Please enter a numeric value.")
    #     for file in top_files_combined:
    #         documents.append(file)
    #     for score in relevance_scores_combined:
    #         cosine_similarity_scores.append(score)
    #     for rating in relevance_ratings:
    #         relevance_labels.append(rating)
    #     # Open file1.py in write mode
    #     with open("trainingdata.py", "w") as file:
    #         # Write the new code to the file
    #         file.write(f"documents = {repr(documents)}\n"
    #                    f"cosine_similarity_scores = {repr(cosine_similarity_scores)}\n"
    #                    f"relevance_labels = {repr(relevance_labels)}\n")
    #     pointwise_ranking(cosine_similarity_scores, relevance_labels)

    ########################
    # Machine Learning Algorithm Training
    # Training Set of 2 Queries

    # Load the saved model from the file
    model_filename = 'pointwise_ranking_model.joblib'
    model = joblib.load(model_filename)

    # Combine new cosine similarity scores into a feature matrix
    X_new = np.array(combined_cosine_similarity).reshape(-1, 1)

    # Predict relevance labels for new query-document pairs
    predicted_relevance_labels = model.predict(X_new)
    # [1.59421771 2.03440386 4.00717956 5.41981466 2.29927175 2.6414384 ]

    # Sort the documents and predicted relevance labels together based on the relevance scores in descending order
    sorted_documents_and_relevance = sorted(
        zip(list_of_file_names, predicted_relevance_labels), key=lambda x: x[1], reverse=True)

    # Weighted scores:
    for i in range(len(combined_cosine_similarity[0])):
        combined_cosine_similarity[0][i] += (0.1 *
                                             predicted_relevance_labels[i])
    sorted_documents_and_weighted_relevance = sorted(
        zip(list_of_file_names, combined_cosine_similarity[0]), key=lambda x: x[1], reverse=True)
    similarities = []
    print("====================================================")
    print(f"Most relevant documents for the query: {query}")
    count = 0
    for doc, relevance in sorted_documents_and_weighted_relevance:
        while count < 3:
            similarities.append(
                {"document": doc, "similarity_score": relevance})
            print(f"== Document: {doc}, Predicted Relevance: {relevance:.2f}")
            count += 1
            break
    return similarities

    # # Extract sentences from top files
    # sentences = dict()
    # # for doc, _ in sorted_documents_and_weighted_relevance:
    # for passage in files[sorted_documents_and_weighted_relevance[0][0]].split("\n"):
    #     for sentence in nltk.sent_tokenize(passage):
    #         tokens = tokenize_and_process(sentence)
    #         if tokens:
    #             sentences[sentence] = tokens

    # # Compute IDF values across sentences
    # idfs = compute_idfs(sentences)

    # # Determine top sentence matches
    # top_matches = top_sentences(querySet, sentences, idfs, n=1)
    # print("The most relevant sentence in the top document is:")
    # for i in range(len(top_matches)):
    #     print(f"== {top_matches[i]}")

    # matches = top_sentences(querySet, sentences, idfs, n=SENTENCE_MATCHES)
    # print("The most relevant sentence(s) in the top documents are:")
    # for i in range(len(matches)):
    #     print(f"== {i+1}: {matches[i]}")


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # query = ["anomaly"]
    # sentences = {"Several learning algorithms aim...": ["Several", "Learning", ...]...}
    # idfs = {"hi": 1.245, "sup": 2.3, ...}
    # n = 1

    # Matching Word Measure
    mwm = {}
    # Query Term Density
    qtd = {}
    # For each word in the query,
    for quer in query:
        # For each sentence and its list of words,
        for sentence, words in sentences.items():
            uniqueWords = []

            # If sentence is not in matching word measurey yet, initialise it
            if not sentence in mwm:
                mwm[sentence] = 0
            # See how many words in it match
            for word in words:
                if (word == quer) and not (quer in uniqueWords):
                    mwm[sentence] += idfs[quer]
                    uniqueWords.append(quer)
    sorted_dict = dict(
        sorted(mwm.items(), key=lambda item: item[1], reverse=True))
    first_key_value = next(iter(sorted_dict.values()))

    # For each top scoring sentence
    for sentence, idfsvalue in sorted_dict.items():
        wordcount = 0
        quercount = 0
        # If same values
        if idfsvalue == first_key_value:
            if not sentence in qtd:
                qtd[sentence] = 0
            for word in sentences[sentence]:
                if word == "====":
                    break
                wordcount += 1
                for quer in query:
                    if quer == word:
                        quercount += 1
            if wordcount != 0:
                # print(sentence, quercount/wordcount)
                qtd[sentence] = quercount/wordcount
        else:
            break

    sorted_keys = sorted(qtd, key=qtd.get, reverse=True)
    return (sorted_keys[:n])


################################################
    # Intrinsic Evaluation of Function
    # reciprocal_ranking(sorted_documents_and_relevance, list_of_file_names)
########################


def pointwise_ranking(cos_sim_scores, rel_lbls):
    """
    The pointwise ranking algorithm takes the `cosine_similarity_scores` of multiple documents,
    along with their `relevance labels`, and trains a linear regression model to predict its
    relevance label based on the cosine_similarity_score.
    """
    # queries = ["I am seeking assistance regarding the migrant workers in my company",
    #            "What do i have to do when i employ new workers"]

    # Combine cosine similarity scores and relevance labels into a feature matrix (X) and target vector (y)
    X = np.array(cos_sim_scores, dtype=object).reshape(-1, 1)
    y = np.array(rel_lbls)

    # Split data into training and testing sets (optional)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Create and train the pointwise ranking model (e.g., Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_filename = 'pointwise_ranking_model.joblib'
    joblib.dump(model, model_filename)


def reciprocal_rank(rank):
    if rank == 0:
        return 0  # If the document is ranked 1st, RR is 1
    return 1 / rank


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    freq = dict()
    dic = dict()
    counter = 0
    # For each file,
    for name, listofwords in documents.items():
        uniqueWords = []
        counter += 1
        for token in listofwords:
            # If this word has not appeared before
            if not token in uniqueWords:
                # If first time appearing
                if (not freq.get(token, 0)):
                    # If the word appears in the document
                    # Increase freq, break into next document
                    freq[token] = 1
                else:
                    # If the word appears in the document
                    # Increase freq, break into next document
                    freq[token] += 1
                uniqueWords.append(token)

    for key, value in freq.items():
        dic[key] = math.log(counter/value)
    # print(dic)
    return dic


if __name__ == '__main__':
    app.run(debug=True)
