import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# ✅ Load Movies Dataset
# ---------------------------
movies = pd.read_csv("movies.csv")   # must contain 'title' & 'genres' columns
movies['genres'] = movies['genres'].fillna("")

# ---------------------------
# ✅ TF-IDF Vectorization (Content Features)
# ---------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_titles = movies['title'].tolist()

# ---------------------------
# ✅ Recommendation Function
# ---------------------------
def recommend_movies(user_rated_movies, user_ratings, top_n=10):
    movie_indices = {title: idx for idx, title in enumerate(movie_titles)}

    # Compute weighted similarity scores
    scores = {}
    for movie, rating in user_ratings.items():
        idx = movie_indices[movie]
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        for i, score in sim_scores:
            if movie_titles[i] in user_rated_movies:
                continue
            scores[movie_titles[i]] = scores.get(movie_titles[i], 0) + (score * rating)

    sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in sorted_movies[:top_n]]

# ---------------------------
# ✅ Streamlit UI
# ---------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Rate movies you’ve seen and get similar movie suggestions!")

selected_movies = st.multiselect(
    "📌 Search & select movies you've watched:",
    options=movie_titles,
    placeholder="Type a movie name..."
)

user_ratings = {}

if selected_movies:
    st.subheader("⭐ Rate Selected Movies")
    for movie in selected_movies:
        user_ratings[movie] = st.slider(f"{movie}", 1, 5, 3)

if st.button("🎯 Get Recommendations"):
    if not user_ratings:
        st.warning("Please rate at least one movie first.")
    else:
        recommendations = recommend_movies(selected_movies, user_ratings)
        
        st.subheader("✅ Recommended Movies for You:")
        for movie in recommendations:
            st.write(f"🎞️ **{movie}**")