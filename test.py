#%% md
# 
#%%
import numpy as np
import pandas as pd
from src.helpers.graphing import plot_movie_factors
from src.utils import vocabulary_based_one_hot_encode, load_pickle

#%%

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
#%%
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

#%%
indexed_data = load_pickle("./indexed_data.pkl")
model = load_pickle("./artifacts/checkpoints/als/1000000000/20250117-073926_lambda0.1_gamma0.1_tau0.1_n_epochs20_n_factors30.pkl")
movie_factors = model.item_factors
#%%
potters = ["4896", "5816", "8368", "40815", "54001", "69844", "81834", "88125", "186777", "267654"]
potters_titles = [item_database[p]["title"] for p in potters]
potters_indices = [indexed_data.id_to_item_bmap.inverse[p] for p in potters]
#%%
plot_movie_factors(movie_factors, potters_indices, potters_titles)