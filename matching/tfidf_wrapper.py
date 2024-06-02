import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_corpus_matrix(corpus):
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    
    # Generate the TF-IDF vectors for the two texts
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return tfidf_matrix, vectorizer

# def tfidf_similarity(query, corpus):
#     # query -  str: "abcdef"
#     # corpus - list: ["def", "ghi"]
#     corpus = [query] + corpus

#     # Create a TfidfVectorizer object
#     vectorizer = TfidfVectorizer()
    
#     # Generate the TF-IDF vectors for the two texts
#     tfidf_matrix = vectorizer.fit_transform(corpus)
    
#     # Calculate the cosine similarity between the two vectors
#     # (1 x 2)
#     cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
#     return cosine_similarities

def get_top_k_similar(sim_matrix, labels, k):

    top_k_indices = sim_matrix.argsort()[-k:][::-1]
    top_k_scores = sim_matrix[top_k_indices]
    top_k_labels = [labels[i] for i in top_k_indices]
    
    return list(zip(top_k_labels, top_k_indices, top_k_scores))

def main(annotation_filepath):
    df = pd.read_json(annotation_filepath, lines=True)
    corpus = df['mistral_instruct_statement'].tolist()
    classes = df['class'].tolist()

    # Temporary query
    query = "What is my name"

    # Prepare corpus
    corpus_matrix, vectorizer = compute_corpus_matrix(corpus)

    # Transform query and retrieve top-k similar records
    query_vector = vectorizer.transform([query])
    sim_matrix = cosine_similarity(query_vector, corpus_matrix).flatten()
    similar_entries = get_top_k_similar(sim_matrix, classes, 4)

    print(similar_entries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Interpretations to Graph")
    parser.add_argument("--annotation_filepath", type=str, required=True)
    args = parser.parse_args()

    main(
        args.annotation_filepath
    )

