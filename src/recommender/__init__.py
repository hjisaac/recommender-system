from abc import abstractmethod, ABC

from main import indexed_data, logger


class Recommender(object):
    def __init__(self, predictor):
        self.predictor = predictor

    def recommend(self, input, fallback_cb):  # to be defined
        predictions = self.predictor.predict(input)
        print("Recommending")
        pass


class AbstractRecommenderBuilder(ABC):

    def __init__(self, backend, *args, **kwargs):
        self.backend = backend

    @abstractmethod
    def build(self):
        pass


class CollaborativeFilteringRecommenderBuilder(AbstractRecommenderBuilder):
    def __init__(self, backend, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, data=indexed_data):
        # Run the model backend code .i.e fit the model and return the prediction capable object
        logger.log(
            f"Starting the build of the recommender using {self.backend.algorithm.__class__.__name__}..."
        )
        logger.log(
            f"Starting a model fitting using the backend {self.backend.algorithm.__class__.__name__}..."
        )
        self.backend(data=data)
        logger.log(
            f"Successfully built the recommender using {self.backend.algorithm.__class__.__name__}"
        )

        return Recommender(predictor=self.backend.model)
