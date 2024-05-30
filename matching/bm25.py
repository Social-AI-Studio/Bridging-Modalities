import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

def bm25_similarity(text1, text2):
    # Preprocessing: Tokenize the texts
    def tokenize(text):
        return text.lower().split()
    
    # Tokenize the texts
    corpus = [tokenize(text1), tokenize(text2)]
    
    # Initialize BM25
    bm25 = BM25Okapi(corpus)
    
    # Generate BM25 vectors for the texts
    bm25_vector1 = bm25.get_scores(tokenize(text1))
    bm25_vector2 = bm25.get_scores(tokenize(text2))
    
    # Calculate cosine similarity between the BM25 vectors
    cosine_sim = cosine_similarity([bm25_vector1], [bm25_vector2])
    
    return cosine_sim[0][0]

# Example usage
# text1 = "This is the first sample text."
# text2 = "This is the second sample text with some different words."
# print(f"Cosine Similarity between the BM25 vectors: {bm25_similarity(text1, text2)}")
