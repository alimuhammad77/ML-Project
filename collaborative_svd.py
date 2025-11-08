# collaborative_svd.py
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
import os

def train_collaborative_svd():
    print("🔹 Training Collaborative Filtering (SVD)...")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    ratings = pd.read_csv("dataset/ratings_100k.csv")
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)

    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"✅ Collaborative Filtering RMSE: {rmse:.4f}")

    # Save the model in models directory
    with open("models/svd_model.pkl", "wb") as f:
        pickle.dump(algo, f)

    return rmse

if __name__ == "__main__":
    train_collaborative_svd()
