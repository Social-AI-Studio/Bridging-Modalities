import os
import argparse
import numpy as np
import pandas as pd
from utils import load_inference_dataset, load_support_dataset
from calc_utils import categories_mapk, label_mapk

def get_top_k_similar(sim_vector, labels, support_target_classes, k, selection):
    if selection == "equal":
        indices = sim_vector.argsort()[::-1]
        
        label_counter = {0: 0, 1: 0}
        count_per_label = k // len(label_counter)
        records = []
        
        for ind in indices:
            label = labels[ind]
            prob = sim_vector[ind]
            target_classes=support_target_classes[ind]

            if label_counter[label] < count_per_label:
                records.append((ind, prob, label, target_classes))
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
        caption_dir,
        sim_matrix_path,
        debug
    ):
    
    print("Loading existing similarity matrix...")
    with open(sim_matrix_path, 'rb') as f:
            sim_matrix = np.load(f)
            labels = np.load(f)
            target_classes = np.load(f, allow_pickle=True)
    
    # Load the inference annotations
    inference_annots = load_inference_dataset(annotation_filepath, caption_dir, None)

    query_labels = []
    query_categories = []
    support_labels = []
    support_categories = []

    if debug:
        inference_annots = inference_annots[:3]
    for index, query in enumerate(inference_annots):

        # Get query labels and categories
        query_labels.append(query['label'])
        if query['label'] == 0:
            query_categories.append([0])
        else:    
            query_categories.append([query['target_categories_mapped']])

        # Retrieve the similarity vectors
        sim_vector = sim_matrix[index]
        similar_entries = get_top_k_similar(sim_vector, labels, target_classes, 16, selection="random")

        # Get support labels and categories
        similar_labels, similar_categories = [], []
        for entry in similar_entries:
            similar_labels.append(entry[2])

            if entry[2] == 0:
                similar_categories.append([0])
            else:    
                similar_categories.append([entry[3]])

        support_labels.append(similar_labels)
        support_categories.append(similar_categories)

    for k in [1,4,8,16]:
        map_at_k = label_mapk(query_labels, support_labels, k, debug)
        print(f"Mean Average Precision at {k}: {map_at_k:04}")

        map_at_k = categories_mapk(query_categories, support_categories, k, debug)
        print(f"Mean Average Precision at {k}: {map_at_k:04}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computing Text Similarity - TF-IDF")
    parser.add_argument("--annotation_filepath", type=str, required=True, help="The zero-shot inference dataset") 
    parser.add_argument("--caption_dir", type=str)
    parser.add_argument("--sim_matrix_filepath", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(
        args.annotation_filepath,
        args.caption_dir,
        args.sim_matrix_filepath,
        args.debug
    )
