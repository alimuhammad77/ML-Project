import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommendation System", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Rate movies and get personalized suggestions!")

# ================= LOAD DATA ==================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    data = pd.merge(ratings, movies, on="movieId")
    return data, movies

data, movies = load_data()

# ================= PIVOT TABLE ==================
user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating").fillna(0)
similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# ================ RECOMMEND FUNCTION =================
def recommend_movies(user_ratings):
    # Append new user rating row at bottom
    df = user_movie_matrix.copy()
    df.loc["new_user"] = 0
    for movie, rating in user_ratings.items():
        df.loc["new_user", movie] = rating
    
    sim = cosine_similarity(df)
    sim_df = pd.DataFrame(sim, index=df.index, columns=df.index)
    
    similar_users = sim_df["new_user"].sort_values(ascending=False)[1:6]

    user_scores = df.loc["new_user"]
    unrated_movies = user_scores[user_scores == 0].index

    scores = {}
    for movie in unrated_movies:
        for user in similar_users.index:
            scores[movie] = scores.get(movie, 0) + df.loc[user, movie] * similar_users[user]

    rec = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return [movie for movie, score in rec if score > 0]

# ================ UI =================
st.subheader("📌 Rate Movies")

movie_choices = movies['title'].unique()[:50]  # first 50 movies
selected_movies = st.multiselect("Select movies you have seen & rate", movie_choices)

user_ratings = {}
for movie in selected_movies:
    rating = st.slider(f"Rate {movie}", 1, 5, 3)
    user_ratings[movie] = rating

if st.button("🎯 Recommend Movies"):
    if len(user_ratings) < 2:
        st.warning("Please rate at least 2 movies ❗")
    else:
        recs = recommend_movies(user_ratings)
        st.subheader("✅ Recommended Movies for You:")
        if not recs:
            st.write("No recommendations found — rate more movies.")
        else:
            for m in recs:
                st.write(f"⭐ {m}")