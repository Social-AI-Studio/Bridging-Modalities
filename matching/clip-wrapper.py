import os
import torch
import clip
import argparse

import pandas as pd


from PIL import Image

def clip_similarity(image_path1, image_path2):

    image1_preprocess = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
    image1_features = model.encode_image(image1_preprocess)

    image2_preprocess = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)
    image2_features = model.encode_image(image2_preprocess)

    similarity = cos(image1_features[0], image2_features[0]).item()
    similarity = (similarity + 1) / 2
    
    return similarity

def main(annotation_filepath, img_folder,  output_folder):
    df = pd.read_json(annotation_filepath, lines=True)
    os.makedirs(output_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for id in df['id']:
        img_filepath = os.path.join(img_folder, f"{id}.jpg")
        img = Image.open(img_filepath) 

        img_preprocess = preprocess(img).unsqueeze(0).to(device)
        img_features = model.encode_image(img_preprocess)
        exit()

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

