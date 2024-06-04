import os
import torch
import clip
import argparse
import pickle
import pandas as pd
import json

from PIL import Image

import os
from utils import load_inference_dataset, load_support_dataset
import numpy as np

def clip_similarity(image_path1, image_path2):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image1_preprocess = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
    image1_features = model.encode_image(image1_preprocess)

    image2_preprocess = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)
    image2_features = model.encode_image(image2_preprocess)

    cos = torch.nn.CosineSimilarity(dim=0)
    similarity = cos(image1_features[0], image2_features[0]).item()
    similarity = (similarity + 1) / 2
    
    return similarity

def clip_corpus_similarity(query_features, corpus_features):

    cos = torch.nn.CosineSimilarity(dim=0)

    sim_vector = []
    for feature in corpus_features:
        similarity = cos(query_features[0], feature[0]).item()
        similarity = (similarity + 1) / 2
        sim_vector.append(similarity)

    return sim_vector

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

        return records
    else:
        indices = sim_vector.argsort()[-k:][::-1]

        records = []
        for ind in indices:
            label = labels[ind]
            prob = sim_vector[ind]

            records.append((ind, prob, label))

        return records
    

def generate_features(annotation_filepath, img_folder,  output_folder):
    # df = pd.read_json(annotation_filepath, lines=True)
    os.makedirs(output_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    data_list = []
    with open(annotation_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            data_list.append(record)

    for record in data_list:
        id = record["id"]
        img_filepath = os.path.join(img_folder, f"{id}.jpg")
        img = Image.open(img_filepath) 

        img_preprocess = preprocess(img).unsqueeze(0).to(device)
        img_features = model.encode_image(img_preprocess)

        # Save img_features as .pkl file
        output_filepath = os.path.join(output_folder, f"{id}.pkl")
        with open(output_filepath, 'wb') as f:
            pickle.dump(img_features, f)

def main(
        annotation_filepath,
        caption_dir,
        feature_dir,
        support_filepaths,
        support_caption_dirs,
        support_feature_dirs,
        output_filepath
    ):
    
    if os.path.exists(output_filepath):
        print("Loading existing similarity matrix...")
        with open(output_filepath, 'rb') as f:
            sim_matrix = np.load(f)
            labels = np.load(f)
    else:
        # Load the inference annotations
        inference_annots = load_inference_dataset(annotation_filepath, caption_dir, feature_dir)
        print("Num Inference Examples:", len(inference_annots))
        
        # Load the support annotations
        support_annots = []
        for filepath, support_caption_dir, support_feature_dir in zip(support_filepaths, support_caption_dirs, support_feature_dirs):
            annots = load_support_dataset(filepath, support_caption_dir, support_feature_dir)
            support_annots += annots
        # corpus = df['mistral_instruct_statement'].tolist()
        # classes = df['class'].tolist()
        print("Num Support Examples:", len(support_annots))

        # Prepare corpus
        corpus = [] 
        labels = []
        for idx, record in enumerate(support_annots):
            if "features" in record.keys():
                corpus.append(record['features'])
                labels.append(record['label'])

        queries = []
        for idx, record in enumerate(inference_annots):
            queries.append(record['features'])

        result = []
        for query in queries:
            sim_vector = clip_corpus_similarity(query, corpus)
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
    parser.add_argument("--feature_dir", type=str, default=None)

    parser.add_argument("--support_filepaths", nargs="+", required=True, help="The support datasets")
    parser.add_argument("--support_caption_dirs", nargs='+')
    parser.add_argument("--support_feature_dirs", nargs='+')

    parser.add_argument("--output_filepath", type=str, required=True, help="The support datasets")
    args = parser.parse_args()

    main(
        args.annotation_filepath,
        args.caption_dir,
        args.feature_dir,
        args.support_filepaths,
        args.support_caption_dirs,
        args.support_feature_dirs,
        args.output_filepath
    )
