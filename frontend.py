import contextlib

import streamlit as st
import math
import requests

import pandas as pd
import itertools
from pprint import pprint
from collections import defaultdict

from src.algorithms.alternating_least_squares import AlternatingLeastSquares
from src.helpers.dataset_indexer import DatasetIndexer
from src.helpers.checkpoint_manager import CheckpointManager
from src.recommenders import CollaborativeFilteringRecommenderBuilder
from src.backends import Backend
from src.helpers._logging import logger  # noqa
from src.settings import settings
from src.utils import vocabulary_based_one_hot_encode

USER_HEADER = "userId"
ITEM_HEADER = "movieId"
RATING_HEADER = "rating"
FEATURE_TO_ENCODE = "genres"
CSV_FILES_DIR = "../../ml-32m/" # The dataset subfolder


# https://files.grouplens.org/datasets/movielens/ml-32m-README.html
ITEM_FEATURE_LIST = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

item_database = (
    pd.read_csv(f"{CSV_FILES_DIR}/movies.csv", dtype={ITEM_HEADER: str})
    .merge(
        pd.read_csv(f"{CSV_FILES_DIR}/links.csv", dtype={ITEM_HEADER: str}),
        on=ITEM_HEADER,
        how="left",
    )
    .assign(
        genres=lambda df: df[FEATURE_TO_ENCODE].apply(
            lambda genres: genres.split("|")
        ),
        features_hot_encoded=lambda df: df[FEATURE_TO_ENCODE].apply(
            lambda g: vocabulary_based_one_hot_encode(
                words=g, vocabulary=ITEM_FEATURE_LIST
            )
        ),
        features_count=lambda df: df["features_hot_encoded"].apply(lambda x: sum(x)),
    )
    .set_index(ITEM_HEADER)  # Set the movieId as the index
    .to_dict(orient="index")  # Convert the DataFrame to a dictionary
)


dataset_indexer = DatasetIndexer(
    # Path to the ratings.csv file
    file_path=f"{CSV_FILES_DIR}/ratings.csv",
    user_header=USER_HEADER,
    item_header=ITEM_HEADER,
    rating_header=RATING_HEADER,
    limit=1_000_000,

    # limit=settings.general.LINES_COUNT_TO_READ,
)

indexed_data = dataset_indexer.index_simple(
    approximate_train_ratio=settings.general.APPROXIMATE_TRAIN_RATIO
)
#%%

als_instance = AlternatingLeastSquares(
    hyper_lambda=settings.als.HYPER_LAMBDA,
    hyper_gamma=settings.als.HYPER_GAMMA,
    hyper_tau=settings.als.HYPER_TAU,
    hyper_n_epochs=settings.als.HYPER_N_EPOCH,
    hyper_n_factors=settings.als.HYPER_N_FACTOR,
)

als_backend = Backend(
    # Define the algorithm
    algorithm=als_instance,
    checkpoint_manager=CheckpointManager(
        checkpoint_folder=settings.als.CHECKPOINT_FOLDER,
        sub_folder=str(settings.general.LINES_COUNT_TO_READ),
    ),
    # The predictor needs this to render the name of the items
    item_database=item_database,
    # Whether we should resume by using the last state of
    # the algorithm the checkpoint manager folder or not.
    resume=True,
    save_checkpoint=False,
)


recommender_builder = CollaborativeFilteringRecommenderBuilder(
    backend=als_backend,
)


recommender = recommender_builder.build(
    data=indexed_data,
    item_database=item_database,
    # Whether to include feature functionality or not
    include_features=True
)


#%%


BASE_URL = "https://image.tmdb.org/t/p/w500"
MOVIES_COUNT = len(indexed_data.id_to_item_bmap)
COLUMNS_PER_ROW = 4


def fetch_movie_from_server(movie_tmdb_id):
    api_key = "63278f37c689ce940e6ded2618abeffe"
    url = f"https://api.themoviedb.org/3/movie/{movie_tmdb_id}?api_key={api_key}"

    try:
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        logger.warning(e)
        raise


def get_movie_poster_url(movie_tmdb_id):
    try:
        data = fetch_movie_from_server(movie_tmdb_id)
        return f"{BASE_URL}{data['poster_path']}"
    except Exception as e:
        logger.error(e)


# Sample movie data
movies_sample = [
    {"id": 1, "title": "Movie 1", "genres": "Action, Adventure", "rating": 4.2},
    {"id": 2, "title": "Movie 2", "genres": "Comedy, Drama", "rating": 3.5},
    # Add more movies here...
]


# Function to filter movies
def filter_movies(movies, search_term, genres):
    return [
        movie
        for movie in movies
        if search_term.lower() in movie["title"].lower()
           and (not genres or any(g.strip() in movie["genres"] for g in genres))
    ]


# Function to render the movie grid
def render_movie_grid():
    rows_count = math.ceil(MOVIES_COUNT / COLUMNS_PER_ROW)

    for row in range(rows_count):
        cols = st.columns(COLUMNS_PER_ROW)
        for i, col in enumerate(cols):
            movie_id = row * COLUMNS_PER_ROW + i
            if movie_id < MOVIES_COUNT:
                movie = indexed_data.id_to_item_bmap[movie_id]  # noqa
                movie_data = item_database[movie]  # noqa
                with col:
                    st.markdown(
                        f"""
                        <div class="movie-container">
                            <h4>{movie_data["title"]}</h4>
                            <p><b>Genres:</b> {" ".join(movie_data["genres"])}</p>
                            <a href="?movie={movie}">
                                <div class="zoom">
                                    <img src={get_movie_poster_url(item_database[movie]["tmdbId"])} style="width: 100%;" alt="Movie Poster">
                                </div>
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# Function to render movie details and rating form
def render_movie_details(movie_id):
    movie_data = item_database[movie_id]

    # Create two columns for side-by-side layout
    col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed

    with col1:

        with contextlib.suppress(Exception):
            remove_data = fetch_movie_from_server(movie_data["tmdbId"])
            st.image(get_movie_poster_url(movie_data["tmdbId"]), width=300)

    with col2:
        # Movie title and description
        st.subheader(movie_data["title"])
        st.write(f"**Genres:** {' '.join(movie_data['genres'])}")
        st.write(f"**Description:** {remove_data['overview']}")  # Assuming you have a 'description' field in your movie_data

        # Rating form
        new_rating = st.number_input("Rate this movie:", min_value=0.0, max_value=5.0, value=5.0, step=0.5)
        if st.button("Go Back to Movie List"):
            st.query_params.clear()

        if st.button("Submit Rating"):
            # Update the movie rating
            logger.info(f"Attempt to compute recommendation using [({movie_id}, {new_rating})]")

            # Call the recommend function
            recommendations = recommend(movie_data, new_rating)

            # Display recommendations
            st.subheader("Based on your rating, here are some recommendations:")
            render_movie_grid()

def recommend(movie, rating):
    # This is a placeholder function. Implement your actual recommendation logic here.
    return ["3", "4", "5"]


# Streamlit Interface
st.title("Movie Recommender ")

# Get query parameters
query_params = st.query_params.to_dict()
movie = query_params.get("movie")

if movie and movie.isdigit():
    # Display movie details and rating form if a movie is clicked
    if movie:
        render_movie_details(movie)
    else:
        st.error("Invalid movie selected.")
else:
    # Filters
    search_term = st.text_input("Search movies...")
    genres = st.multiselect(
        "Filter by genres",
        options=list(set(g.strip() for movie in movies_sample for g in movie["genres"].split(","))),
    )

    # Filter movies based on input
    filtered_movies = filter_movies(movies_sample, search_term, genres)

    # Add custom CSS for the zoom effect
    st.markdown("""
        <style>
            .zoom:hover {
                transform: scale(1.2);
                transition: transform 0.3s ease;
            }
            .movie-container {
                text-align: center;
                padding: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

    # Show filtered movies or a "no results" message
    if not filtered_movies:
        st.warning("No movies found. Try adjusting your search or filters.")
    else:
        # render_movie_grid(filtered_movies, columns_per_row)
        render_movie_grid()
