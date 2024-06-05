# import required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import pickle 
import json
from utils import load_inference_dataset, load_support_dataset
import numpy as np


def sift_similarity(image_feat1, image_feat2, bf_matcher):
    matches = bf_matcher.knnMatch(image_feat1,image_feat2, k=2)
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
        else:
            continue
        if m.distance < 0.1*n.distance:
            good_matches.append([m])

    return len(good_matches)

def sift_corpus_similarity(query_features, corpus_features):

    bf = cv2.BFMatcher()
    sim_vector = []
    count =0
    for feature in corpus_features:
        if feature is None:
            count += 1
            continue
        similarity = sift_similarity(query_features, feature, bf)
        sim_vector.append(similarity)
    return sim_vector

def get_top_k_similar(sim_vector, labels, k, selection):
    print(sim_vector)
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
    

def generate_features(annotation_filepath, img_folder,  output_folder):
    # df = pd.read_json(annotation_filepath, lines=True)
    os.makedirs(output_folder, exist_ok=True)

    data_list = []
    with open(annotation_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            data_list.append(record)

    for record in data_list:
        id = str(record["id"])
        img_filepath = os.path.join(img_folder, f"{id}.jpg")

        img = cv2.imread(img_filepath, 0)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        # Save img_features as .pkl file
        output_filepath = os.path.join(output_folder, f"{id}.pkl")
        with open(output_filepath, 'wb') as f:
            pickle.dump(descriptors, f)

def main(
        annotation_filepath,
        caption_dir,
        feature_dir,
        image_dir, 
        support_filepaths,
        support_caption_dirs,
        support_feature_dirs,
        support_image_dirs,
        output_filepath
    ):
    
    if os.path.exists(output_filepath):
        print("Loading existing similarity matrix...")
        with open(output_filepath, 'rb') as f:
            sim_matrix = np.load(f)
            labels = np.load(f)
    else:
        # Load the inference annotations
        if not os.path.exists(feature_dir):
            print("Inference features not generated. Generating...")
            # generate features for them 
            generate_features(annotation_filepath, image_dir, feature_dir)
        inference_annots = load_inference_dataset(annotation_filepath, caption_dir, feature_dir)
        print("Num Inference Examples:", len(inference_annots))
        
        # Load the support annotations
        support_annots = []
        for filepath, support_caption_dir, support_feature_dir, support_image_dir, in zip(support_filepaths, support_caption_dirs, support_feature_dirs, support_image_dirs):
            if not os.path.exists(support_feature_dir) and support_feature_dir != "":
                # generate features for them 
                print("Support features not generated. Generating...")
                generate_features(filepath, support_image_dir, support_feature_dir)
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

        sim_matrix = []
        for idx, query in enumerate(queries):
            print(idx)
            sim_vector = sift_corpus_similarity(query, corpus)
            sim_matrix.append(sim_vector)

        with open(output_filepath, 'wb') as f:
            np.save(f, np.array(sim_matrix))
            np.save(f, np.array(labels))

        sim_matrix = np.array(sim_matrix)
        labels = np.array(labels)
    # Example: Getting top 4 similar records for first record
    sim_vector = sim_matrix[0]
    similar_entries = get_top_k_similar(sim_vector, labels, 4, selection="random")
    print(similar_entries)

    sim_vector = sim_matrix[0]
    similar_entries = get_top_k_similar(sim_vector, labels, 4, selection="equal")
    print(similar_entries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computing Text Similarity - CLIP")
    parser.add_argument("--annotation_filepath", type=str, required=True, help="The zero-shot inference dataset") 
    parser.add_argument("--caption_dir", type=str, default=None)
    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)

    parser.add_argument("--support_filepaths", nargs="+", required=True, help="The support datasets")
    parser.add_argument("--support_caption_dirs", nargs='+')
    parser.add_argument("--support_feature_dirs", nargs='+')
    parser.add_argument("--support_image_dirs", nargs='+')

    parser.add_argument("--output_filepath", type=str, required=True, help="The support datasets")
    args = parser.parse_args()

    main(
        args.annotation_filepath,
        args.caption_dir,
        args.feature_dir,
        args.image_dir,
        args.support_filepaths,
        args.support_caption_dirs,
        args.support_feature_dirs,
        args.support_image_dirs,
        args.output_filepath
    )