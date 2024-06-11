import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_inference_dataset, load_support_dataset


def compute_corpus_matrix(corpus):
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    
    # Generate the TF-IDF vectors for the two texts
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return tfidf_matrix, vectorizer

def get_top_k_similar(sim_vector, labels, support_target_classes, k, selection):
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
            target_classes=support_target_classes[ind]

            records.append((ind, prob, label, target_classes))

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

    print("Annotation (Inference) Content:", annotation_content)
    print("Support Content:", support_content)
    
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
        
        corpus_matrix, vectorizer = compute_corpus_matrix(corpus)
        print("Corpus Matrix:", corpus_matrix.shape)

        # Prepare inference queries
        queries = []
        for idx, record in enumerate(inference_annots):
            queries.append(record[annotation_content])

        query_vectors = vectorizer.transform(queries)
        print("Query Vectors:", query_vectors.shape)

        # Compute cosine similarity
        sim_matrix = cosine_similarity(query_vectors, corpus_matrix)
        print("Similarity Matrices:", sim_matrix.shape)
        
        with open(output_filepath, 'wb') as f:
            np.save(f, sim_matrix)
            np.save(f, np.array(labels))
            np.save(f, np.array(target_classes, dtype=object))

    # Example: Getting top 4 similar records for first record
    sim_vector = sim_matrix[0]
    similar_entries = get_top_k_similar(sim_vector, labels, target_classes, 6, selection="random")
    print(similar_entries)

    sim_vector = sim_matrix[0]
    similar_entries = get_top_k_similar(sim_vector, labels, target_classes, 6, selection="equal")
    print(similar_entries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computing Text Similarity - TF-IDF")
    parser.add_argument("--annotation_filepath", type=str, required=True, help="The zero-shot inference dataset") 
    parser.add_argument("--annotation_content", type=str, required=True, choices=["content_text", "content_text_caption"])
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
