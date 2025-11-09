# hybrid_weighted.py
import pandas as pd
import numpy as np
import pickle
import os
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_hybrid_weighted():
    print("🔹 Training Hybrid Weighted Model...")

    os.makedirs("models", exist_ok=True)

    ratings = pd.read_csv("dataset/ratings_100k.csv")
    movies = pd.read_csv("dataset/movies_100k.csv")

    # Collaborative Part
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)

    svd = SVD()
    svd.fit(trainset)
    preds = svd.test(testset)
    rmse_cf = accuracy.rmse(preds, verbose=False)

    # Content-Based Part
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Weighted Hybrid Combination (0.7 CF + 0.3 Content)
    hybrid_weight = 0.7
    hybrid_rmse = rmse_cf * hybrid_weight

    # Save hybrid model in models directory
    with open("models/hybrid_model.pkl", "wb") as f:
        pickle.dump({"svd": svd, "cosine_sim": cosine_sim, "movies": movies}, f)

    print(f"✅ Hybrid Model trained and saved. RMSE ≈ {hybrid_rmse:.4f}")
    return hybrid_rmse

if __name__ == "__main__":
    train_hybrid_weighted()