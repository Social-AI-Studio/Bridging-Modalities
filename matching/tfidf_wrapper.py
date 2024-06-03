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

def get_top_k_similar(sim_matrix, labels, k):

    top_k_indices = sim_matrix.argsort()[-k:][::-1]
    
    if labels == []:
        return top_k_indices
    
    all_indices = sim_matrix.argsort()[::-1]
    label_counter = {0: 0, 1: 0}
    equal_indices = []
    target_count = k // len(label_counter)

    for index in all_indices:
        label = labels[index]
        if label_counter[label] < target_count:
            equal_indices.append(index)
            label_counter[label] += 1
        if len(equal_indices) == k:
            break
    return equal_indices

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
