import os
import argparse
import numpy as np
import pandas as pd
from utils import load_inference_dataset, load_support_dataset
import random
from calc_utils import mapk

def main(
        annotation_filepath,
        caption_dir,
        support_filepaths,
        support_caption_dirs,
        top_p
    ):
    
    inference_annots = load_inference_dataset(annotation_filepath, caption_dir, None)
    print("Num Inference Examples:", len(inference_annots))
    
    # Load the support annotations
    support_annots = []
    for filepath, support_caption_dir in zip(support_filepaths, support_caption_dirs):
        annots = load_support_dataset(filepath, support_caption_dir, None)
        support_annots += annots

    print("Num Support Examples:", len(support_annots))

    total_queries_classes = []
    total_support_classes = []

    for index, query in enumerate(inference_annots):

        total_queries_classes.append([query['target_categories_mapped']])
        samples = random.sample(support_annots, top_p)
        
        retrieved_doc_classes = [record['target_categories_mapped'] for record in samples]
        total_support_classes.append(retrieved_doc_classes)

    map_at_k = mapk(total_queries_classes, total_support_classes, top_p)
    print(f"Mean Average Precision at {top_p}: {map_at_k:04}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computing Text Similarity - TF-IDF")
    parser.add_argument("--annotation_filepath", type=str, required=True, help="The zero-shot inference dataset") 
    parser.add_argument("--caption_dir", type=str)
    parser.add_argument("--support_filepaths", nargs="+", required=True, help="The support datasets")
    parser.add_argument("--support_caption_dirs", nargs='+')
    parser.add_argument("--top_p", type=int)
    args = parser.parse_args()

    main(
        args.annotation_filepath,
        args.caption_dir,
        args.support_filepaths,
        args.support_caption_dirs,
        args.top_p
    )
