import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

print("📥 Loading dataset...")

# Load MovieLens data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge movie names with ratings
data = pd.merge(ratings, movies, on="movieId")

print("✅ Dataset loaded and merged.")

# Create User-Movie Rating Matrix
print("📊 Creating user-movie matrix...")
user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating").fillna(0)

# Compute User-User similarity
print("🤖 Training model (calculating user similarity)...")
similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

print("✅ Model training complete!")

# Save model + matrix
print("💾 Saving model files...")
pickle.dump(user_movie_matrix, open("user_movie_matrix.pkl", "wb"))
pickle.dump(similarity_df, open("user_similarity.pkl", "wb"))

print("🎉 Training completed & files saved:")
print("   - user_movie_matrix.pkl")
print("   - user_similarity.pkl")