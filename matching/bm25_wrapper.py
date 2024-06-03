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

    query = "What is my name"
    sim_matrix = bm25_similarity(query, corpus)
    similar_entries = get_top_k_similar(sim_matrix, classes, 4)
    print(similar_entries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Interpretations to Graph")
    parser.add_argument("--annotation_filepath", type=str, required=True)
    args = parser.parse_args()

    main(args.annotation_filepath)
