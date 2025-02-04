import os
import streamlit as st
import requests
import pandas as pd
import time
from dotenv import load_dotenv

from src.algorithms.alternating_least_squares import AlternatingLeastSquares
from src.helpers.dataset_indexer import DatasetIndexer
from src.helpers.checkpoint_manager import CheckpointManager
from src.recommenders import CollaborativeFilteringRecommenderBuilder
from src.backends import Backend
from src.helpers._logging import logger # noqa
from src.settings import settings
from src.utils import vocabulary_based_one_hot_encode, save_pickle, load_pickle

# Load env variables
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Constants
USER_HEADER = "userId"
ITEM_HEADER = "movieId"
RATING_HEADER = "rating"
FEATURE_TO_ENCODE = "genres"
CSV_FILES_DIR = "./datasets/movielens"  # The dataset subfolder
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
    item_database = (  # noqa
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
    logger.info("Item database loaded successfully.")
    return item_database


# Cache the indexed data
@st.cache_resource
def index_data():
    try:
        indexed_data = load_pickle("indexed_data.pkl")
        logger.info("indexed_data loaded successfully.")
        return indexed_data
    except Exception as e:
        logger.error(e)
        dataset_indexer = DatasetIndexer(
            file_path=f"{CSV_FILES_DIR}/ratings.csv",
            user_header=USER_HEADER,
            item_header=ITEM_HEADER,
            rating_header=RATING_HEADER,
            limit=1_000_000_000,
        )
        indexed_data = dataset_indexer.index_simple( # noqa
            approximate_train_ratio=settings.general.APPROXIMATE_TRAIN_RATIO
        )
        save_pickle(indexed_data, "indexed_data.pkl")
        logger.info("`indexed_data` has been pickled successfully.")
        return indexed_data


# Cache the recommender model
@st.cache_resource
def get_recommender(_indexed_data, item_database): # noqa
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
        logger.info("`recommender` object has been pickled successfully.")
        return recommender


# Load data and models
item_database = load_item_database()
indexed_data = index_data()
recommender = get_recommender(indexed_data, item_database)


# Function to fetch movie details from TMDB
def fetch_movie_from_server(movie_tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_tmdb_id}?api_key={TMDB_API_KEY}"
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


def render_movie_details(movie_data, image_width=300, double_column=True):
    remote_data = fetch_movie_from_server(movie_data["tmdbId"])
    try:
        poster = get_movie_poster_url(movie_data["tmdbId"])
    except Exception as e:  # noqa
        logger.error(e)
        poster = None

    if not double_column:
        st.subheader(movie_data["title"])
        st.write(f"**Genres:** {' '.join(movie_data['genres'])}")
        if poster is not None:
            st.image(poster, width=image_width)
        return

    # Create two columns for side-by-side layout
    col1, col2 = st.columns([1, 3])  # Adjust the ratio as needed

    with col1:
        if poster is not None:
            st.image(poster, width=image_width)

    with col2:
        # Movie title and description
        st.subheader(movie_data["title"])
        st.write(f"**Genres:** {' '.join(movie_data['genres'])}")

        st.write(
            f"**Description:** {remote_data.get('overview')}"
        )  # Assuming you have a 'description' field in your movie_data


def filter_movies(search_term, max_results=10):  # noqa
    search_term = search_term.lower()  # noqa
    filtered_movies = []  # noqa

    # Iterate over the items in the database
    for k, v in item_database.items():
        # If the title matches the search term, add it to the list
        if search_term in v["title"].lower():
            # The presence of the feature_hot_encoded np.array is making streamlit to crash so
            # remove it for the meantime.
            movie_data = {"id": k, **v}
            if "features_hot_encoded" in movie_data:
                movie_data["features_hot_encoded"] = movie_data["features_hot_encoded"].tolist()
            filtered_movies.append(movie_data)
        # Stop when we reach the desired number of results
        if len(filtered_movies) >= max_results:
            break
    return filtered_movies


# Interface
st.title("Movie Recommender")

selected_movie = None
search_term = st.text_input("Search for a movie...")
# Delay debounce logic to prevent frequent re-rendering during typing
if search_term:
    # Filter movies based on the search term
    start_time = time.time()
    filtered_movies = filter_movies(search_term)

    end_time = time.time()
    execution_time = end_time - start_time
    st.write(f"Execution Time for filter_movies: {execution_time:.4f} seconds")
    if filtered_movies:
        selected_movie = st.selectbox(
            "Select a movie", options=filtered_movies, format_func=lambda x: x["title"]
        )

        # Display the selected movie details
        st.subheader(f"Selected Movie: {selected_movie['title']}")
        st.write(f"Genres: {', '.join(selected_movie['genres'])}")

        # Step 4: Allow user to rate the movie
        user_rating = st.slider(
            "Rate this movie", min_value=0.0, max_value=5.0, value=5.0, step=0.5
        )


# Step 5: Display recommendations based on the rating
if st.button("Submit Rating"):
    if selected_movie:
        # Call the recommend method of the recommender object
        recommender_input = [(selected_movie["id"], user_rating)]
        recommendations = recommender.recommend(recommender_input)

        # Step 6: Display recommended movies
        st.subheader("Recommended Movies:")
        for r in recommendations:
            render_movie_details(r, double_column=True, image_width=100)
            # st.write(f"- {r['title']} ({', '.join(r['genres'])})")
    else:
        st.warning("Seach for a movie a rate it to get recommendations")
