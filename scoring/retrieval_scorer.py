import os
import argparse
import numpy as np
import pandas as pd
from utils import load_inference_dataset, load_support_dataset
from calc_utils import categories_mapk, label_mapk

def get_top_k_similar(sim_vector, support_annots, k, demonstration_selection):

    if demonstration_selection == "random":
        indices = sim_vector[:k]
    else:
        indices = sim_vector.argsort()[-k:][::-1]

    records = []
    for ind in indices:
        annot = support_annots[ind]

        records.append((ind, 0, annot['label'], annot['target_categories_mapped']))

    return records

def main(
        annotation_filepath,
        caption_dir,
        support_filepaths,
        support_caption_dirs,
        support_feature_dirs,
        demonstration_selection,
        sim_matrix_path,
        debug
    ):
    
    print("Loading existing similarity matrix...")
    with open(sim_matrix_path, 'rb') as f:
            sim_matrix = np.load(f)        
    print("Similarity Matrix:", sim_matrix.shape)
    
    # Load the inference annotations
    inference_annots = load_inference_dataset(annotation_filepath, caption_dir, None)
    print("Inference Annots:", len(inference_annots))

    support_annots = []
    for filepath, support_caption_dir, support_feature_dir in zip(support_filepaths, support_caption_dirs, support_feature_dirs):
        annots = load_support_dataset(filepath, support_caption_dir, support_feature_dir)
        support_annots += annots
    print("Support Annots:", len(support_annots))

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
            query_categories.append(query['target_categories_mapped'])

        # Retrieve the similarity vectors
        sim_vector = sim_matrix[index]
        similar_entries = get_top_k_similar(sim_vector, support_annots, 16, demonstration_selection)

        # Get support labels and categories
        similar_labels, similar_categories = [], []
        for entry in similar_entries:
            similar_labels.append(entry[2])

            if entry[2] == 0:
                similar_categories.append([0])
            else:    
                similar_categories.append(entry[3])

        support_labels.append(similar_labels)
        support_categories.append(similar_categories)

    print("Computing for Binary Labels...")
    maps = []
    for k in [1,4,8,16]:
        map_at_k = label_mapk(query_labels, support_labels, k, debug)
        print(f"Mean Average Precision at {k}: {map_at_k:04}")
        map_at_k = str(round(map_at_k, 4))
        maps.append(map_at_k)
    print(",".join(maps))

    print("Computing for Fine-Grained Categories...")
    maps = []
    for k in [1,4,8,16]:
        map_at_k = categories_mapk(query_categories, support_categories, k, debug)
        print(f"Mean Average Precision at {k}: {map_at_k:04}")
        map_at_k = str(round(map_at_k, 4))
        maps.append(map_at_k)
    print(",".join(maps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computing Text Similarity - TF-IDF")
    parser.add_argument("--annotation_filepath", type=str, required=True, help="The zero-shot inference dataset") 
    parser.add_argument("--caption_dir", type=str)
    parser.add_argument("--support_filepaths", nargs='+')
    parser.add_argument("--support_caption_dirs", nargs='+')
    parser.add_argument("--support_feature_dirs", nargs='+')
    parser.add_argument("--demonstration_selection", type=str, required=True)
    parser.add_argument("--sim_matrix_filepath", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(
        args.annotation_filepath,
        args.caption_dir,
        args.support_filepaths,
        args.support_caption_dirs,
        args.support_feature_dirs,
        args.demonstration_selection,
        args.sim_matrix_filepath,
        args.debug
    )
