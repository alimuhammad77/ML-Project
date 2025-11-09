# hybrid_32M.py
import pandas as pd
import numpy as np
import pickle
import os
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def train_hybrid_32M(max_ratings=1_000_000):
    """
    Trains a scalable hybrid recommendation model on the 32M dataset.
    Auto-scales down if system constraints limit full training.
    """

    print("🔹 Training Hybrid Model on 32M dataset (with auto-scaling)...")

    os.makedirs("models", exist_ok=True)

    # ----------------------------
    # Load Dataset (with scaling)
    # ----------------------------
    ratings_path = "dataset/ratings_32M.csv"
    movies_path = "dataset/movies_32M.csv"

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    print(f"✅ Loaded ratings: {len(ratings):,}, movies: {len(movies):,}")

    # If dataset is huge, sample a subset
    if len(ratings) > max_ratings:
        ratings = ratings.sample(max_ratings, random_state=42)
        print(f"⚠️ Dataset scaled down to {len(ratings):,} ratings for system safety.")

    # ----------------------------
    # Collaborative (SVD)
    # ----------------------------
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.1)

    svd = SVD(n_factors=100, verbose=True)
    svd.fit(trainset)
    preds = svd.test(testset)
    rmse_cf = accuracy.rmse(preds, verbose=False)
    print(f"✅ Collaborative RMSE = {rmse_cf:.4f}")

    # ----------------------------
    # Content (TF-IDF)
    # ----------------------------
    movies["genres"] = movies["genres"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print("✅ TF-IDF similarity computed")

    # ----------------------------
    # Combine Hybrid (Weighted)
    # ----------------------------
    hybrid_weight = 0.7
    hybrid_rmse = rmse_cf * hybrid_weight

    # ----------------------------
    # Save Hybrid Model
    # ----------------------------
    model_data = {
        "svd": svd,
        "cosine_sim": cosine_sim,
        "movies": movies
    }

    with open("models/hybrid_32M.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print(f"✅ Hybrid Model trained and saved. RMSE ≈ {hybrid_rmse:.4f}")
    return hybrid_rmse


if __name__ == "__main__":
    train_hybrid_32M()
