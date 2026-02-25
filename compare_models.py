from collaborative_svd import train_collaborative_svd
from content_tfidf import train_content_tfidf
from hybrid_weighted import train_hybrid_weighted
import os

print("🔹 Comparing Models...\n")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

rmse_svd = train_collaborative_svd()
train_content_tfidf()
rmse_hybrid = train_hybrid_weighted()

print("\n📈 Model Comparison Results:")
print(f"1️⃣ Collaborative Filtering (SVD): RMSE = {rmse_svd:.4f}")
print(f"2️⃣ Content-Based Filtering (TF-IDF): N/A (similarity-based)")
print(f"3️⃣ Hybrid Weighted: RMSE ≈ {rmse_hybrid:.4f}")

if rmse_hybrid < rmse_svd:
    print("\n🏆 Selected Model: Hybrid Weighted (Best Performance)")
else:
    print("\n🏆 Selected Model: Collaborative SVD (Best Performance)")
