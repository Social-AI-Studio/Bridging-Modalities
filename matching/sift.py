# import required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

def sift_similarity(image_path1, image_path2):

    # read two input images as grayscale
    img1 = cv2.imread(image_path1, 0)
    img2 = cv2.imread(image_path2, 0)

    # Check if images are loaded properly
    if img1 is None or img2 is None:
        raise ValueError("One of the images didn't load properly. Check the paths.")

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # detect and compute the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1,des2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.1*n.distance:
            good_matches.append([m])

    return len(good_matches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Interpretations to Graph")
    parser.add_argument("--img_path1", type=str, required=True)
    parser.add_argument("--img_path2", type=str, required=True)
    
    args = parser.parse_args()

    num_matches = sift_similarity(args.img_path1, args.img_path2)
    print(f"Number of matches between the images: {num_matches}")