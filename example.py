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

from src.helpers.graphing import (
    plot_als_train_test_loss_evolution,
    plot_als_train_test_rmse_evolution,
    # plot_error_evolution,
    plot_power_low_distribution,
    plot_data_item_distribution_as_hist,
)


# In[2]:


dataset_indexer = DatasetIndexer(
    file_path="./ml-32m/ratings.csv",
    user_header="userId",
    item_header="movieId",
    rating_header="rating",
    limit=settings.general.LINES_COUNT_TO_READ,
)

indexed_data = dataset_indexer.index_simple(
    approximate_train_ratio=settings.general.APPROXIMATE_TRAIN_RATIO
)


# In[3]:


# Import the movie csv file that will act as our movie database
# And that database is needed by the backend to query the movies
item_database = (
    pd.read_csv("./ml-32m/movies.csv", dtype={"movieId": str})
    .set_index("movieId")
    .to_dict(orient="index")
)


# In[4]:


# plot_data_item_distribution_as_hist(indexed_data)


# In[5]:


# plot_power_low_distribution(indexed_data,)


# In[6]:


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
    resume=False,
    save_checkpoint=True,
)


# In[7]:


recommender_builder = CollaborativeFilteringRecommenderBuilder(
    backend=als_backend,
)

# This might take some moment before finishing
recommender = recommender_builder.build(data=indexed_data)


# In[ ]:





# In[8]:


# plot_als_train_test_rmse_evolution(als_backend.algorithm)


# In[11]:


# plot_als_train_test_loss_evolution(als_backend.algorithm)


# In[10]:


#
prediction_input = [("17", 4)]
recommender.recommend(prediction_input)


# In[15]:


prediction_input = [("267654", 4)]  # Harry Poter
recommender.recommend(prediction_input)


# In[ ]:




