import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")

# -------- Load Data --------
@st.cache_data
def load_movies():
    return pd.read_csv("dataset/movies_100k.csv")

@st.cache_data
def load_ratings():
    return pd.read_csv("dataset/ratings_100k.csv")

movies = load_movies()
ratings = load_ratings()

title_to_id = dict(zip(movies['title'], movies['movieId']))
id_to_title = dict(zip(movies['movieId'], movies['title']))

# -------- Load Models --------
def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def load_npy(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None

svd_model = load_model("models/svd_model.pkl")
content_sim = load_npy("models/content_similarity.npy")

# -------- Recommendation Functions --------
def recommend_cf(user_ratings_dict, top_n=10):
    if svd_model is None:
        st.error("❌ SVD model not found. Train using collaborative_svd.py")
        return pd.DataFrame()
    all_movie_ids = movies['movieId'].unique()
    rated_ids = list(user_ratings_dict.keys())
    candidates = [m for m in all_movie_ids if m not in rated_ids]
    predictions = []
    for mid in candidates[:1000]:
        preds = [svd_model.predict(uid, mid).est for uid in ratings['userId'].unique()[:20]]
        predictions.append((mid, np.mean(preds)))
    df = pd.DataFrame(predictions, columns=['movieId', 'score'])
    return df.merge(movies, on='movieId').sort_values('score', ascending=False).head(top_n)

def recommend_content(user_ratings_dict, top_n=10):
    if content_sim is None:
        st.error("❌ Content similarity not found. Train using content_tfidf.py")
        return pd.DataFrame()
    id_to_idx = {mid: idx for idx, mid in enumerate(movies['movieId'])}
    rated = [mid for mid in user_ratings_dict if mid in id_to_idx]
    if not rated:
        st.warning("⚠️ Please rate at least one movie.")
        return pd.DataFrame()
    user_profile = np.zeros(len(movies))
    for mid, rating in user_ratings_dict.items():
        idx = id_to_idx[mid]
        user_profile += content_sim[idx] * rating
    scores = [(movies.iloc[i]['movieId'], score) for i, score in enumerate(user_profile)]
    df = pd.DataFrame(scores, columns=['movieId', 'score'])
    df = df[~df['movieId'].isin(rated)]
    return df.merge(movies, on='movieId').sort_values('score', ascending=False).head(top_n)

def recommend_hybrid(user_ratings_dict, top_n=10, w_cf=0.6, w_cb=0.4):
    cf = recommend_cf(user_ratings_dict, top_n=None)
    cb = recommend_content(user_ratings_dict, top_n=None)
    cf_scores = dict(zip(cf['movieId'], cf['score']))
    cb_scores = dict(zip(cb['movieId'], cb['score']))
    all_ids = set(cf_scores.keys()).union(cb_scores.keys())
    combined = [(mid, w_cf * cf_scores.get(mid, 3.0) + w_cb * cb_scores.get(mid, 0.0)) for mid in all_ids]
    df = pd.DataFrame(combined, columns=['movieId', 'score'])
    return df.merge(movies, on='movieId').sort_values('score', ascending=False).head(top_n)

# -------- Streamlit UI --------
st.title("🎥 Movie Recommendation System")
st.write("Input movies you've watched and rate them to get personalized recommendations.")

model_choice = st.selectbox(
    "Select a recommendation model:",
    ["Collaborative Filtering (SVD)", "Content-Based (TF-IDF)", "Hybrid Model"]
)

if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# --- Search bar with live movie suggestions ---
search_query = st.text_input("🔍 Search a movie:")

suggested_movies = (
    movies[movies['title'].str.contains(search_query, case=False, na=False)]
    if search_query
    else pd.DataFrame()
)

if not suggested_movies.empty:
    selected_movie = st.selectbox(
        "🎬 Select a movie from search results:",
        suggested_movies['title'].head(15).tolist()
    )
else:
    selected_movie = None

movie_rating = st.slider("⭐ Your Rating", 0.5, 5.0, 3.0, 0.5)

col1, col2 = st.columns(2)
with col1:
    if st.button("➕ Add Movie"):
        if selected_movie:
            movie_id = title_to_id[selected_movie]
            st.session_state.user_ratings[movie_id] = movie_rating
            st.success(f"✅ Added: {selected_movie} - {movie_rating}/5")
        else:
            st.error("❌ Please select a valid movie from search results.")
with col2:
    if st.button("🗑️ Remove Selected Movie"):
        if selected_movie:
            movie_id = title_to_id.get(selected_movie)
            if movie_id in st.session_state.user_ratings:
                del st.session_state.user_ratings[movie_id]
                st.success(f"🗑️ Removed: {selected_movie}")
            else:
                st.warning("⚠️ Movie not in your rated list.")
        else:
            st.error("❌ Please select a movie to remove.")

# --- Display rated movies ---
if st.session_state.user_ratings:
    st.markdown("### 🎬 Your Rated Movies")
    rated_df = pd.DataFrame([
        {"Title": id_to_title[mid], "Rating": rating}
        for mid, rating in st.session_state.user_ratings.items()
    ])
    st.table(rated_df)

if st.button("🎯 Get Recommendations"):
    if not st.session_state.user_ratings:
        st.warning("⚠️ Please add at least one movie before getting recommendations.")
    else:
        user_ratings_dict = st.session_state.user_ratings
        if model_choice == "Collaborative Filtering (SVD)":
            recs = recommend_cf(user_ratings_dict)
        elif model_choice == "Content-Based (TF-IDF)":
            recs = recommend_content(user_ratings_dict)
        else:
            recs = recommend_hybrid(user_ratings_dict)

        if recs.empty:
            st.warning("⚠️ No recommendations found. Try rating more movies.")
        else:
            st.success("🎉 Recommended Movies:")
            st.dataframe(recs[['title', 'genres', 'score']].reset_index(drop=True))

st.markdown("---")
st.caption("Developed by Ali Mohammad — Movie Recommendation Project")
