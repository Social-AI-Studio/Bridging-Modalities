import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k_similar(sim_matrix, labels, k):
    top_k_similar = []
    
    for idx, label in zip(range(sim_matrix.shape[0]), labels):
        similar_indices = sim_matrix[idx].argsort()[-(k+1):-1][::-1]
        similar_scores = sim_matrix[idx][similar_indices]
        top_k_similar.append(list(zip(labels, similar_indices, similar_scores)))
    
    return top_k_similar

def tfidf_similarity(corpus):
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    
    # Generate the TF-IDF vectors for the two texts
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate the cosine similarity between the two vectors
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def main(annotation_filepath):
    df = pd.read_json(annotation_filepath, lines=True)
    corpus = df['mistral_instruct_statement'].tolist()
    classes = df['class'].tolist()

    sim_matrix = tfidf_similarity(corpus)
    sim_matrix = get_top_k_similar(sim_matrix, classes, 4)
    print(sim_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Interpretations to Graph")
    parser.add_argument("--annotation_filepath", type=str, required=True)
    args = parser.parse_args()

    main(
        args.annotation_filepath
    )

