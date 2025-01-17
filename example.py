#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from src.algorithms.alternating_least_squares import AlternatingLeastSquares
from src.helpers.dataset_indexer import DatasetIndexer
from src.helpers.checkpoint_manager import CheckpointManager
from src.recommenders import CollaborativeFilteringRecommenderBuilder
from src.backends import Backend
from src.helpers._logging import logger  # noqa
from src.settings import settings
from src.utils import vocabulary_based_one_hot_encode, load_pickle, save_pickle

from src.helpers.graphing import (
    plot_als_train_test_loss_evolution,
    plot_als_train_test_rmse_evolution,
    # plot_error_evolution,
    plot_power_low_distribution,
    plot_data_item_distribution_as_hist,
)


# In[2]:


USER_HEADER = "userId"
ITEM_HEADER = "movieId"
RATING_HEADER = "rating"
FEATURE_TO_ENCODE = "genres"
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


# In[3]:


dataset_indexer = DatasetIndexer(
    file_path="./ml-32m/ratings.csv",
    user_header=USER_HEADER,
    item_header=ITEM_HEADER,
    rating_header=RATING_HEADER,
    limit=settings.general.LINES_COUNT_TO_READ,
)

indexed_data = dataset_indexer.index_simple(
    approximate_train_ratio=settings.general.APPROXIMATE_TRAIN_RATIO
)


# In[4]:


# Import the movies csv file joined with the movie links csv file and that will act
# as our movie database. The backend needs this database to query the movies.
item_database = (
    pd.read_csv("./ml-32m/movies.csv", dtype={ITEM_HEADER: str})
    .merge(
        pd.read_csv("./ml-32m/links.csv", dtype={ITEM_HEADER: str}),
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


# In[5]:


# plot_data_item_distribution_as_hist(indexed_data)


# In[6]:


# plot_power_low_distribution(indexed_data,)


# In[7]:


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


# In[8]:


recommender_builder = CollaborativeFilteringRecommenderBuilder(
    backend=als_backend,
)

# This might take some moment before finishing
recommender = recommender_builder.build(
    data=indexed_data, item_database=item_database, include_features=True
)


# In[ ]:





# In[9]:


# plot_als_train_test_rmse_evolution(als_backend.algorithm)


# In[10]:


# plot_als_train_test_loss_evolution(als_backend.algorithm)


# In[11]:


#
prediction_input = [("17", 4)]
recommender.recommend(prediction_input)


# In[12]:


prediction_input = [("267654", 4)]  # Harry Poter
recommender.recommend(prediction_input)


# In[13]:


#
# recommender.recommend()


# In[ ]:




