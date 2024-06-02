import argparse
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

def bm25_similarity(query, corpus):
    # Preprocessing: Tokenize the texts
    def tokenize(text):
        return text.lower().split()
    corpus = [query] + corpus
    # Tokenize the entire corpus
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    
    # Initialize BM25
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Tokenize the query
    tokenized_query = tokenize(query)
    
    # Generate BM25 scores for the query against the corpus
    sim_scores = bm25.get_scores(tokenized_query)
    return sim_scores[1:]

def get_top_k_similar(sim_matrix, labels, k):
    top_k_indices = sim_matrix.argsort()[-k:][::-1]
    top_k_scores = sim_matrix[top_k_indices]
    top_k_labels = [labels[i] for i in top_k_indices]
    
    return top_k_indices

def main(annotation_filepath):
    df = pd.read_json(annotation_filepath, lines=True)
    corpus = df['mistral_instruct_statement'].tolist()
    classes = df['class'].tolist()

    query = "What is my name"
    sim_matrix = bm25_similarity(query, corpus)
    similar_entries = get_top_k_similar(sim_matrix, classes, 4)
    print(similar_entries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Interpretations to Graph")
    parser.add_argument("--annotation_filepath", type=str, required=True)
    args = parser.parse_args()

    main(args.annotation_filepath)
