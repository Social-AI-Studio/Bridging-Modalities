import argparse
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import os
from utils import load_inference_dataset, load_support_dataset

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

def get_top_k_similar(sim_vector, labels, k, selection):
    if selection == "equal":
        indices = sim_vector.argsort()[::-1]
        
        label_counter = {0: 0, 1: 0}
        count_per_label = k // len(label_counter)
        records = []
        
        for ind in indices:
            label = labels[ind]
            prob = sim_vector[ind]

            if label_counter[label] < count_per_label:
                records.append((ind, prob, label))
                label_counter[label] += 1

            if len(records) == k:
                break

        records_sorted_by_label = sorted(records, key=lambda x: x[2], reverse=True)
        return records_sorted_by_label
    else:
        indices = sim_vector.argsort()[-k:][::-1]

        records = []
        for ind in indices:
            label = labels[ind]
            prob = sim_vector[ind]

            records.append((ind, prob, label))

        return records


def main(
        annotation_filepath,
        caption_dir,
        support_filepaths,
        support_caption_dirs,
        output_filepath
    ):
    
    if os.path.exists(output_filepath):
        print("Loading existing similarity matrix...")
        with open(output_filepath, 'rb') as f:
            sim_matrix = np.load(f)
            labels = np.load(f)
    else:
        # Load the inference annotations
        inference_annots = load_inference_dataset(annotation_filepath, caption_dir, None)
        print("Num Inference Examples:", len(inference_annots))
        
        # Load the support annotations
        support_annots = []
        for filepath, support_caption_dir in zip(support_filepaths, support_caption_dirs):
            annots = load_support_dataset(filepath, support_caption_dir, None)
            support_annots += annots
        # corpus = df['mistral_instruct_statement'].tolist()
        # classes = df['class'].tolist()
        print("Num Support Examples:", len(support_annots))

        # Prepare corpus
        corpus = [] 
        labels = []
        for idx, record in enumerate(support_annots):
            corpus.append(record['content_for_retrieval'])
            labels.append(record['label'])

        queries = []
        for idx, record in enumerate(inference_annots):
            queries.append(record['content_for_retrieval'])

        result = []
        for query in queries:
            sim_vector = bm25_similarity(query, corpus)
            result.append(sim_vector)

        with open(output_filepath, 'wb') as f:
            np.save(f, np.array(result))
            np.save(f, np.array(labels))

    # Example: Getting top 4 similar records for first record
    sim_vector = sim_matrix[0]
    similar_entries = get_top_k_similar(sim_vector, labels, 4, selection="random")
    print(similar_entries)

    sim_vector = sim_matrix[0]
    similar_entries = get_top_k_similar(sim_vector, labels, 4, selection="equal")
    print(similar_entries)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computing Text Similarity - TF-IDF")
    parser.add_argument("--annotation_filepath", type=str, required=True, help="The zero-shot inference dataset") 
    parser.add_argument("--caption_dir", type=str, default=None)

    parser.add_argument("--support_filepaths", nargs="+", required=True, help="The support datasets")
    parser.add_argument("--support_caption_dirs", nargs='+')

    parser.add_argument("--output_filepath", type=str, required=True, help="The support datasets")
    args = parser.parse_args()

    main(
        args.annotation_filepath,
        args.caption_dir,
        args.support_filepaths,
        args.support_caption_dirs,
        args.output_filepath
    )
