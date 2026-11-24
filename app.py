
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Page Configuration
st.set_page_config(
    page_title="CineMind | Movie Intelligence",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #0F172A;
    color: white;
}
h1, h2, h3 {
    color: #38BDF8;
}
.movie-card {
    background: #1E293B;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# Prepare Content-Based Model
@st.cache_resource
def build_content_model(genres):
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(genres.fillna("Unknown"))
    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )
    model.fit(matrix)
    return model, matrix

content_model, tfidf_matrix = build_content_model(
    movies["genres"].str.replace("|", " ", regex=False)
)

# Prepare Collaborative Model
@st.cache_resource
def build_user_model(ratings):
    matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )
    filled = matrix.fillna(0)
    similarity = cosine_similarity(filled)
    return matrix, similarity

user_matrix, user_similarity = build_user_model(ratings)

# Content Recommendation
def recommend_content(movie_name, number=10):

    matches = movies[
        movies["title"].str.contains(
            movie_name, case=False, na=False, regex=False
        )
    ]

    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]

    distances, indices = content_model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=min(number + 1, len(movies))
    )

    result = movies.iloc[indices[0][1:]].copy()

    return result.head(number)

# Collaborative Recommendation
def recommend_collaborative(user_id, number=10):

    if user_id not in user_matrix.index:
        return pd.DataFrame()

    user_index = list(user_matrix.index).index(user_id)

    similar_users = np.argsort(
        user_similarity[user_index]
    )[::-1][1:11]

    similar_ids = user_matrix.index[similar_users]

    watched = user_matrix.loc[user_id].dropna().index

    scores = user_matrix.loc[similar_ids].mean()

    scores = scores.drop(labels=watched, errors="ignore")

    top_ids = scores.sort_values(
        ascending=False
    ).head(number).index

    return movies[movies["movieId"].isin(top_ids)]

# Website Header
st.markdown("""
<div style="text-align:center; padding:25px;">
    <h2 style="color:#38BDF8;">
        Developed by Aditya Kumar
    </h2>
    <p style="color:#94A3B8;">
        BCA (Artificial Intelligence and Machine Learning)
    </p>
</div>
""", unsafe_allow_html=True)
st.title("🎬 CineMind")
st.subheader("Your Personal Movie Intelligence")
st.write("Discover movies based on your taste and preferences.")

st.divider()

# Sidebar
st.sidebar.title("🎥 CineMind Menu")

page = st.sidebar.radio(
    "Choose Section",
    [
        "Home",
        "Movie Recommendation",
        "Collaborative Recommendation",
        "About Project"
    ]
)

if page == "Home":

    st.header("Welcome to CineMind 🚀")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", len(movies))
    col2.metric("Total Ratings", len(ratings))
    col3.metric("Total Users", ratings["userId"].nunique())

    st.subheader("Popular Movies")

    popular = ratings.groupby("movieId").agg(
        total_ratings=("rating", "count"),
        average_rating=("rating", "mean")
    ).reset_index()

    popular = popular.merge(movies, on="movieId")

    popular = popular.sort_values(
        "total_ratings", ascending=False
    ).head(10)

    st.dataframe(
        popular[["title", "genres", "average_rating", "total_ratings"]],
        use_container_width=True
    )

elif page == "Movie Recommendation":

    st.header("🎯 Find Similar Movies")

    movie_name = st.text_input(
        "Enter Movie Name",
        placeholder="Example: Toy Story"
    )

    if st.button("Recommend Movies"):

        if movie_name.strip():

            result = recommend_content(movie_name, 10)

            if result.empty:
                st.warning("Movie not found. Try another movie.")
            else:
                st.success("Recommended Movies")
                st.dataframe(
                    result[["title", "genres"]],
                    use_container_width=True
                )

        else:
            st.warning("Please enter a movie name.")

elif page == "Collaborative Recommendation":

    st.header("🤖 User-Based Recommendations")

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        max_value=int(user_matrix.index.max()),
        value=1
    )

    if st.button("Get Recommendations"):

        result = recommend_collaborative(int(user_id), 10)

        if result.empty:
            st.warning("No recommendations available.")
        else:
            st.success("Movies recommended for you!")
            st.dataframe(
                result[["title", "genres"]],
                use_container_width=True
            )

elif page == "About Project":

    st.header("About CineMind")

    st.write("""
    CineMind is a Personalized Movie Intelligence System.

    It uses Machine Learning techniques to recommend movies.

    **Technologies Used:**
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit

    **Recommendation Techniques:**
    - Content-Based Filtering
    - Collaborative Filtering
    """)
