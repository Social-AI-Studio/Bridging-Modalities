# import required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import pickle 
import json

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

def get_top_k_similar(sim_matrix, labels, k):
    import numpy as np
    sim_matrix = np.array(sim_matrix)
    top_k_indices = sim_matrix.argsort()[-k:][::-1]
    return top_k_indices

def main(annotation_filepath, img_folder,  output_folder):
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

        print(img_filepath)
        img = cv2.imread(img_filepath, 0)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        # Save img_features as .pkl file
        output_filepath = os.path.join(output_folder, f"{id}.pkl")
        with open(output_filepath, 'wb') as f:
            pickle.dump(descriptors, f)

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