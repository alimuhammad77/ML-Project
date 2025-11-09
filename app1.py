import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------------------------
# ✅ Load Data
# ---------------------------------
@st.cache_data
def load_movies():
    return pd.read_csv("dataset/movies_100k.csv")

movies = load_movies()
movie_titles = movies["title"].tolist()
movie_ids = dict(zip(movies["title"], movies["movieId"]))
id_to_title = dict(zip(movies["movieId"], movies["title"]))

# ---------------------------------
# ✅ Load Models
# ---------------------------------
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_npy(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None

svd_model = load_model("models/svd_model.pkl")
content_sim = load_npy("models/content_similarity.npy")
hybrid_model = load_model("models/hybrid_model.pkl")

# ---------------------------------
# ✅ Recommendation Functions
# ---------------------------------
def recommend_cf(user_ratings_dict, top_n=10):
    if svd_model is None:
        st.error("❌ Collaborative model not found.")
        return []
    all_movies = movies["movieId"].tolist()
    rated = set(user_ratings_dict.keys())
    candidates = [m for m in all_movies if m not in rated]

    preds = []
    for mid in candidates[:2000]:
        est = np.mean([svd_model.predict(uid, mid).est for uid in range(1, 11)])
        preds.append((mid, est))
    sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [id_to_title[mid] for mid, _ in sorted_preds]

def recommend_content(user_ratings_dict, top_n=10):
    if content_sim is None:
        st.error("❌ Content model not found.")
        return []
    id_to_idx = {m: i for i, m in enumerate(movies["movieId"])}
    rated = [m for m in user_ratings_dict if m in id_to_idx]
    if not rated:
        return []

    user_profile = np.zeros(len(movies))
    for mid, rating in user_ratings_dict.items():
        idx = id_to_idx[mid]
        user_profile += content_sim[idx] * rating

    scores = [(movies.iloc[i]["movieId"], s) for i, s in enumerate(user_profile)]
    df = pd.DataFrame(scores, columns=["movieId", "score"])
    df = df[~df["movieId"].isin(rated)]
    df = df.sort_values("score", ascending=False).head(top_n)
    return [id_to_title[mid] for mid in df["movieId"].tolist()]

def recommend_hybrid(user_ratings_dict, top_n=10, w_cf=0.6, w_cb=0.4):
    if svd_model is None or content_sim is None:
        st.error("❌ Hybrid model components missing.")
        return []
    cf_list = recommend_cf(user_ratings_dict, top_n=None)
    cb_list = recommend_content(user_ratings_dict, top_n=None)

    cf_scores = {title: i for i, title in enumerate(cf_list[::-1])}
    cb_scores = {title: i for i, title in enumerate(cb_list[::-1])}
    all_titles = set(cf_scores.keys()) | set(cb_scores.keys())

    combined = []
    for title in all_titles:
        score = w_cf * cf_scores.get(title, 0) + w_cb * cb_scores.get(title, 0)
        combined.append((title, score))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    return [t for t, _ in sorted_combined[:top_n]]

# ---------------------------------
# ✅ Streamlit UI
# ---------------------------------
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="centered")
st.title("🎬 Movie Recommendation System")
st.write("Rate movies you’ve seen and get personalized recommendations!")

# Session State
if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# Model selection
model_choice = st.selectbox(
    "📊 Choose a Recommendation Model:",
    ["Collaborative Filtering (SVD)", "Content-Based (TF-IDF)", "Hybrid"]
)

# Movie search & add
search_query = st.text_input("🔍 Search for a movie:", placeholder="Type a movie name...")
if search_query:
    matches = [m for m in movie_titles if search_query.lower() in m.lower()]
    if matches:
        selected = st.selectbox("Select from results:", matches)
        if st.button("➕ Add Movie"):
            if selected not in st.session_state.selected_movies:
                st.session_state.selected_movies.append(selected)
                st.session_state.user_ratings[movie_ids[selected]] = 3
            else:
                st.warning("Movie already added.")
    else:
        st.warning("No movies found matching your search.")

# Show selected movies with ratings + remove option
if st.session_state.selected_movies:
    st.subheader("⭐ Your Rated Movies")
    for title in st.session_state.selected_movies:
        cols = st.columns([4, 1])
        with cols[0]:
            rating = st.slider(title, 1, 5, st.session_state.user_ratings[movie_ids[title]])
            st.session_state.user_ratings[movie_ids[title]] = rating
        with cols[1]:
            if st.button("❌ Remove", key=f"remove_{title}"):
                st.session_state.selected_movies.remove(title)
                del st.session_state.user_ratings[movie_ids[title]]
                st.experimental_rerun()

# Recommendations
if st.button("🎯 Get Recommendations"):
    if not st.session_state.user_ratings:
        st.warning("Please rate at least one movie.")
    else:
        user_ratings_dict = st.session_state.user_ratings
        if model_choice == "Collaborative Filtering (SVD)":
            recs = recommend_cf(user_ratings_dict)
        elif model_choice == "Content-Based (TF-IDF)":
            recs = recommend_content(user_ratings_dict)
        else:
            recs = recommend_hybrid(user_ratings_dict)

        if not recs:
            st.warning("No recommendations found.")
        else:
            st.subheader("✅ Recommended Movies:")
            for movie in recs:
                st.write(f"🎞️ **{movie}**")