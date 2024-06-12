import os
import tqdm

import argparse
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from utils import load_inference_dataset, load_support_dataset

def tokenize(text):
    return text.lower().split()

def bm25_similarity(query, bm25):
    # Tokenize the query
    tokenized_query = tokenize(query)
    
    # Generate BM25 scores for the query against the corpus
    sim_scores = bm25.get_scores(tokenized_query)
    return sim_scores

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
        annotation_content,
        caption_dir,
        support_filepaths,
        support_caption_dirs,
        support_content,
        output_filepath,
        overwrite
    ):
    
    sim_matrix = []
    print("Annotation (Inference) Content:", annotation_content)
    print("Support Content:", support_content)
    print()
    
    if os.path.exists(output_filepath) and not overwrite:
        print("Loading existing similarity matrix...")
        with open(output_filepath, 'rb') as f:
            sim_matrix = np.load(f)
            labels = np.load(f)
            target_classes=np.load(f)
    else:
        # Load the inference annotations
        inference_annots = load_inference_dataset(annotation_filepath, caption_dir, None)
        print("Num Inference Examples:", len(inference_annots))
        
        # Load the support annotations
        support_annots = []
        for filepath, support_caption_dir in zip(support_filepaths, support_caption_dirs):
            annots = load_support_dataset(filepath, support_caption_dir, None)
            support_annots += annots

        print("Num Support Examples:", len(support_annots))

        # Prepare corpus
        corpus = [] 
        labels = []
        target_classes = []
        for idx, record in enumerate(support_annots):
            corpus.append(record[support_content])
            labels.append(record['label'])
            target_classes.append(record['target_categories_mapped'])

        
        # Preprocessing: Tokenize the texts
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        # Prepare inference queries
        for record in tqdm.tqdm(inference_annots):
            query = record[annotation_content]
            sim_vector = bm25_similarity(query, bm25)
            sim_matrix.append(sim_vector)

        with open(output_filepath, 'wb') as f:
            np.save(f, np.array(sim_matrix))
            np.save(f, np.array(labels))
            np.save(f, np.array(target_classes, dtype=object))

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
    parser.add_argument("--annotation_content", type=str, required=True, choices=["content_text", "content_text_caption", "rationale"])
    parser.add_argument("--caption_dir", type=str, default=None)

    parser.add_argument("--support_filepaths", nargs="+", required=True, help="The support datasets")
    parser.add_argument("--support_caption_dirs", nargs='+')
    parser.add_argument("--support_content", type=str, required=True, choices=["content_text", "content_text_caption", "rationale"])

    parser.add_argument("--output_filepath", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(
        args.annotation_filepath,
        args.annotation_content,
        args.caption_dir,
        args.support_filepaths,
        args.support_caption_dirs,
        args.support_content,
        args.output_filepath,
        args.overwrite
    )
