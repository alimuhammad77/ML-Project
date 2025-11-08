# content_tfidf.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import numpy as np
import os

def train_content_tfidf():
    print("🔹 Training Content-Based Filtering (TF-IDF)...")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    movies = pd.read_csv("dataset/movies_100k.csv")
    if "genres" not in movies.columns:
        raise ValueError("❌ movies_100k.csv must include a 'genres' column.")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Save model in models directory
    np.save("models/content_similarity.npy", cosine_sim)

    print("✅ Content-Based model trained and saved successfully.")
    return cosine_sim

if __name__ == "__main__":
    train_content_tfidf()
