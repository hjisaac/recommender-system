from abc import abstractmethod, ABC

from src.backends import Backend
from src.helpers._logging import logger  # noqa


class Recommender(object):
    def __init__(self, predictor):
        self.predictor = predictor

    def recommend(self, input_data=None):  # noqa
        predictions = self.predictor.predict(input_data)
        return self.predictor.render(predictions)


class AbstractRecommenderBuilder(ABC):

    def __init__(self, backend: Backend, *args, **kwargs):
        self.backend = backend

    @abstractmethod
    def build(self):
        pass


class CollaborativeFilteringRecommenderBuilder(AbstractRecommenderBuilder):
    def __init__(self, backend: Backend, *args, **kwargs):
        super().__init__(backend, *args, **kwargs)

    def build(self, data):
        # Run the model backend code .i.e fit the model and return the prediction-capable object
        logger.info(
            f"Starting the build of the recommender using {self.backend.algorithm.__class__.__name__} "
            f"with the state {self.backend.algorithm.state.to_dict()}"
        )
        logger.info(
            f"Starting a model fitting using the backend {self.backend.algorithm.__class__.__name__}..."
        )
        predictor = self.backend(data=data)
        logger.info(
            f"Successfully built the recommender using {self.backend.algorithm.__class__.__name__}"
        )

        return Recommender(predictor=predictor)
