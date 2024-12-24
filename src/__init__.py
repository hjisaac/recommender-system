import logging
from src.utils.data_loader import DataLoader
from .backends.alternating_least_squares import AlternatingLeastSquares


logger = logging.getLogger(__name__)


class RecommenderBuilder(object):
    def __init__(self, data_loader: DataLoader, backend: AlternatingLeastSquares):
        self.data_loader = data_loader
        self.model_backend = backend
        pass

    def build(self, *args, **kwargs):
        self.data_loader.load()
        # This is where the training is happening
        self.model_backend.train(self.data_loader)


recommender_builder = RecommenderBuilder(
    DataLoader(),
    AlternatingLeastSquares(),
)

recommender = recommender_builder.build()
recommender.recommend()
