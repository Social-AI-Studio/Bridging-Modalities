import os
import torch
import clip
import argparse
import pickle
import pandas as pd
import json

from PIL import Image

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

def get_top_k_similar(sim_matrix, labels, k):
    import numpy as np
    sim_matrix = np.array(sim_matrix)
    top_k_indices = sim_matrix.argsort()[-k:][::-1]
    return top_k_indices

def main(annotation_filepath, img_folder,  output_folder):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Interpretations to Graph")
    parser.add_argument("--annotation_filepath", type=str, required=True)
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    main(
        args.annotation_filepath,
        args.img_folder,
        args.output_folder
    )

