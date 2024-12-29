#!/usr/bin/env python
# coding: utf-8

# In[9]:


from src.algorithms.alternating_least_squares import AlternatingLeastSquares
from src.helpers.dataset_indexer import DatasetIndexer
from src.helpers.checkpoint_manager import CheckpointManager
from src.recommenders import CollaborativeFilteringRecommenderBuilder
from src.backends import Backend

from src.settings import settings


# In[10]:


dataset_indexer = DatasetIndexer(
    file_path="./ml-32m/ratings.csv",
    user_header="userId",
    item_header="movieId",
    rating_header="rating",
    limit=10000,
)

indexed_data = dataset_indexer.index(approximate_train_ratio=0.8)


# In[11]:


alternating_least_squares = AlternatingLeastSquares(
    hyper_lambda=settings.als.HYPER_LAMBDA,
    hyper_gamma=settings.als.HYPER_GAMMA,
    hyper_tau=settings.als.HYPER_TAU,
    hyper_n_epochs=settings.als.HYPER_N_EPOCH,
    hyper_n_factors=settings.als.HYPER_N_FACTOR,
)

als_backend = Backend(
    # Define the algorithm
    algorithm=alternating_least_squares,
    checkpoint_manager=CheckpointManager(
        checkpoint_folder=settings.als.CHECKPOINT_FOLDER,
    ),
)


# In[12]:


recommender_builder = CollaborativeFilteringRecommenderBuilder(
    backend=als_backend,
)

recommender = recommender_builder.build(data=indexed_data)

# recommenders.recommend(None)


# In[23]:




