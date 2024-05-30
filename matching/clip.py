import torch
import clip
from PIL import Image

def clip_similarity(image_path1, image_path2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    cos = torch.nn.CosineSimilarity(dim=0)

    image1_preprocess = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
    image1_features = model.encode_image(image1_preprocess)

    image2_preprocess = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)
    image2_features = model.encode_image(image2_preprocess)

    similarity = cos(image1_features[0], image2_features[0]).item()
    similarity = (similarity + 1) / 2
    
    return similarity

# Example usage
# image1 = "car.jpeg"
# image2 = "car256.jpeg"
# print(f"Image Similarity: {clip_similarity(image1, image2)}")