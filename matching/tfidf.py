from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_similarity(text1, text2):
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    
    # Generate the TF-IDF vectors for the two texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate the cosine similarity between the two vectors
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return cosine_sim[0][0]

# Example usage
# text1 = "The quick brown fox jumps over the lazy dog"
# text2 = "A fast brown fox leaps over a sleepy dog"
# print(f"Text Similarity: {tfidf_similarity(text1, text2)}")
