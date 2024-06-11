import os
import argparse
import numpy as np
import pandas as pd
from utils import load_inference_dataset, load_support_dataset

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

def count_relevant(query_list, doc_lists):
    result = 0
    for doc_list in doc_lists:
        intersection = set(query_list) & set(doc_list)
        if intersection:
            result += 1
    return result

def main(
        annotation_filepath,
        caption_dir,
        sim_matrix_path,
        top_p
    ):
    
    print("Loading existing similarity matrix...")
    with open(sim_matrix_path, 'rb') as f:
            sim_matrix = np.load(f)
            labels = np.load(f)
            target_classes=np.load(f, allow_pickle=True)
    
    # Load the inference annotations
    inference_annots = load_inference_dataset(annotation_filepath, caption_dir, None)

    total_intersections = 0
    for index, query in enumerate(inference_annots):

        query_classes = query['target_categories_mapped']
        sim_vector = sim_matrix[index]
        similar_entries = get_top_k_similar(sim_vector, labels, target_classes, top_p, selection="random")
        retrieved_doc_classes = [entry[3] for entry in similar_entries]
        
        intersections = count_relevant(query_classes, retrieved_doc_classes)
        total_intersections += intersections

    precision_at_k = total_intersections / (len(inference_annots) * top_p)
    print(f"Precision at {top_p}: {precision_at_k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computing Text Similarity - TF-IDF")
    parser.add_argument("--annotation_filepath", type=str, required=True, help="The zero-shot inference dataset") 
    parser.add_argument("--caption_dir", type=str)
    parser.add_argument("--sim_matrix_filepath", type=str)
    parser.add_argument("--top_p", type=int)
    args = parser.parse_args()

    main(
        args.annotation_filepath,
        args.caption_dir,
        args.sim_matrix_filepath,
        args.top_p
    )
