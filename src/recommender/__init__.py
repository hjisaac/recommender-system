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

    def __init__(self, model_backend, *args, **kwargs):
        self.model_backend = model_backend

    @abstractmethod
    def build(self):
        pass


class CollaborativeFilteringRecommenderBuilder(AbstractRecommenderBuilder):
    def __init__(self, backend, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, data=indexed_data):
        # Run the model backend code .i.e fit the model and return the prediction capable object
        logger.log(
            f"Starting the build of the recommender using {self.model_backend.__class__.__name__}..."
        )
        logger.log(
            f"Starting a model fitting using the backend {self.model_backend.__class__.__name__}..."
        )
        self.model_backend.fit_model(data=indexed_data)
        logger.log(
            f"Successfully built the recommender using {self.model_backend.__class__.__name__}"
        )

        return Recommender(predictor=self.model_backend.model)
