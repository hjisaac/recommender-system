import contextlib
import streamlit as st
import math
import requests
import pandas as pd
from src.algorithms.alternating_least_squares import AlternatingLeastSquares
from src.helpers.dataset_indexer import DatasetIndexer
from src.helpers.checkpoint_manager import CheckpointManager
from src.recommenders import CollaborativeFilteringRecommenderBuilder
from src.backends import Backend
from src.helpers._logging import logger
from src.settings import settings
from src.utils import vocabulary_based_one_hot_encode, save_pickle, load_pickle

# Constants
USER_HEADER = "userId"
ITEM_HEADER = "movieId"
RATING_HEADER = "rating"
FEATURE_TO_ENCODE = "genres"
CSV_FILES_DIR = "ml-32m/"  # The dataset subfolder
BASE_URL = "https://image.tmdb.org/t/p/w500"
TOTAL_MOVIES_COUNT = 10000  # Adjust based on your dataset
COLUMNS_PER_ROW = 4

# Genre list for one-hot encoding
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


# Cache the item database
@st.cache_resource
def load_item_database():
    print("Calling load_item_database")
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
            features_count=lambda df: df["features_hot_encoded"].apply(
                lambda x: sum(x)
            ),
        )
        .set_index(ITEM_HEADER)
        .to_dict(orient="index")
    )
    return item_database


# Cache the indexed data
@st.cache_resource
def index_data():
    print("Calling index_data")
    try:
        print("unpickling")
        return load_pickle("indexed_data.pkl")
    except Exception as e:
        print("Pickling")
        logger.error(e)
        dataset_indexer = DatasetIndexer(
            file_path=f"{CSV_FILES_DIR}/ratings.csv",
            user_header=USER_HEADER,
            item_header=ITEM_HEADER,
            rating_header=RATING_HEADER,
            limit=1_000_000,
        )
        indexed_data = dataset_indexer.index_simple(
            approximate_train_ratio=settings.general.APPROXIMATE_TRAIN_RATIO
        )
        save_pickle(indexed_data, "indexed_data.pkl.backup")
        return indexed_data


# Cache the recommender model
@st.cache_resource
def get_recommender(_indexed_data, item_database):
    try:
        return load_pickle("./recommender.pkl")
    except Exception as e:
        logger.error(e)
        als_instance = AlternatingLeastSquares(
            hyper_lambda=settings.als.HYPER_LAMBDA,
            hyper_gamma=settings.als.HYPER_GAMMA,
            hyper_tau=settings.als.HYPER_TAU,
            hyper_n_epochs=settings.als.HYPER_N_EPOCH,
            hyper_n_factors=settings.als.HYPER_N_FACTOR,
        )
        als_backend = Backend(
            algorithm=als_instance,
            checkpoint_manager=CheckpointManager(
                checkpoint_folder=settings.als.CHECKPOINT_FOLDER,
                sub_folder=str(settings.general.LINES_COUNT_TO_READ),
            ),
            item_database=item_database,
            resume=True,
            save_checkpoint=False,
        )
        recommender_builder = CollaborativeFilteringRecommenderBuilder(
            backend=als_backend,
        )
        recommender = recommender_builder.build(
            data=_indexed_data,
            item_database=item_database,
            include_features=True,
        )
        save_pickle(recommender, "./recommender.pkl")
        return recommender


# Load data and models
item_database = load_item_database()
indexed_data = index_data()
recommender = get_recommender(indexed_data, item_database)


# Function to fetch movie details from TMDB
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


# Function to get movie poster URL
def get_movie_poster_url(movie_tmdb_id):
    try:
        data = fetch_movie_from_server(movie_tmdb_id)
        return f"{BASE_URL}{data['poster_path']}"
    except Exception as e:
        logger.error(e)
        return None


# Function to render the movie grid
def render_movie_grid(to_show=None):
    movies_limit = len(to_show) if to_show else TOTAL_MOVIES_COUNT
    rows_count = math.ceil(movies_limit / COLUMNS_PER_ROW)

    for row in range(rows_count):
        cols = st.columns(COLUMNS_PER_ROW)
        for i, col in enumerate(cols):
            movie_id = row * COLUMNS_PER_ROW + i
            if to_show:
                movie_id = to_show[movie_id]
            if movie_id < movies_limit:
                movie = indexed_data.id_to_item_bmap[movie_id]
                movie_data = item_database[movie]
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


# Streamlit Interface
st.title("Movie Recommender")

# Initialize session state for selected movie and recommendations
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

# Create two columns
col1, col2 = st.columns([1, 2])

# Left column: Search and rate movies
with col1:
    st.subheader("Search and Rate Movies")
    search_term = st.text_input("Search movies...")
    genres = st.multiselect(
        "Filter by genres",
        options=ITEM_FEATURE_LIST,  # Use the genre list for filtering
    )

    # Render the movie grid
    render_movie_grid()

    # If a movie is selected, show its details and rating form
    if st.session_state.selected_movie:
        movie_data = item_database[st.session_state.selected_movie]
        st.subheader(movie_data["title"])
        st.write(f"**Genres:** {' '.join(movie_data['genres'])}")
        remove_data = fetch_movie_from_server(movie_data["tmdbId"])
        st.write(f"**Description:** {remove_data['overview']}")

        new_rating = st.number_input(
            "Rate this movie:", min_value=0.0, max_value=5.0, value=5.0, step=0.5
        )
        if st.button("Submit Rating"):
            # Update recommendations
            recommender_input = [(st.session_state.selected_movie, new_rating)]
            st.session_state.recommendations = recommender.recommend(recommender_input)
            st.session_state.selected_movie = None  # Clear selection after rating

# Right column: Show recommendations
with col2:
    st.subheader("Recommendations")
    if st.session_state.recommendations:
        render_movie_grid([r["item_id"] for r in st.session_state.recommendations])
    else:
        st.write("Rate a movie to see recommendations!")
