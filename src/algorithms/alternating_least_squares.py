import sys
import numpy as np

from enum import Enum
from tqdm import tqdm
from typing import Optional
from functools import cached_property
from types import NoneType

from src.algorithms.core import Algorithm
from src.helpers.dataset_indexer import IndexedDatasetWrapper
from src.helpers.state_manager import AlgorithmState
from src.helpers.serial_mapper import (
    SerialUnidirectionalMapper,
    SerialBidirectionalMapper,
)
from src.helpers.predictor import Predictor
from src.helpers._logging import logger  # noqa
from src.settings import settings


_als_cache = {}


def _clear_als_cache():  # noqa
    global _als_cache
    _als_cache = {}


# Centralize this if needed somewhere else
class LearningTargetEnum(str, Enum):
    USER = "user"
    ITEM = "item"

    # No need to cache this method because the caching overhead
    # will compensate the 0(2) of the direct method call.
    @classmethod
    def targets(cls):
        """
        Returns the list of the entries' values
        """
        return [member.value for member in cls]


class AlternatingLeastSquaresState(AlgorithmState):

    # We still need the algorithm in other to do prediction

    @staticmethod
    def to_predictor(
        als: "AlternatingLeastSquares", *args, item_database=None, **kwargs
    ) -> Predictor:

        def validate_ratings_data(user_ratings_data):
            """
            Validates the rating data by checking if the provided item is known.

            Parameters:
            - user_ratings_data: List of tuples containing item IDs and their corresponding ratings.
            - id_to_item_bmap: Dictionary mapping item IDs to item details.

            Returns:
            - A list of valid (item_id, rating) tuples where the item_id exists in id_to_item_bmap.
            """

            if not user_ratings_data:
                return

            valid_ratings = []
            invalid_ratings = []
            for item, rating in user_ratings_data:
                if als.id_to_item_bmap.inverse[item] is not None:
                    valid_ratings.append((item, rating))
                else:
                    invalid_ratings.append((item, rating))
            if invalid_ratings:
                logger.error(
                    f"The provided user ratings data contains the following unknown item rating(s), skipping unknown items' ratings {invalid_ratings}"
                )
            return valid_ratings

        def predict(user_ratings_data: list = None):
            """
            Predict ratings for a user based on user and item factors and biases
            and his historical ratings' data.

            Args:
                user_ratings_data (list): User's historical ratings' data.

            Returns:
                np.ndarray: Predicted ratings for all items.
            """

            user_ratings_data = validate_ratings_data(user_ratings_data)

            if not user_ratings_data:
                # Return the averaged rating prediction for all the items
                return np.mean(
                    # Matrix product .i.e the equivalent of the "@" operator
                    np.dot(als.user_factors, als.item_factors.T)
                    + als.user_biases.reshape(-1, 1)
                    + als.item_biases.reshape(1, -1),
                    axis=0,
                )

            user_factor = None

            for _ in range(als.PREDICTION_EPOCHS):
                user_factor, user_bias = als.learn_user_bias_and_factor(
                    user_id=None,
                    user_factor=user_factor,
                    user_ratings_data=user_ratings_data,
                )

            # The order of the vectors in the matrix product matters as they have
            # the following shape respectively: (`items_count`, hyper_n_factors)
            # and (hyper_n_factors, 1). Broadcasting is used for the biases' additions
            return np.dot(user_factor, als.item_factors.T) + user_bias + als.item_biases

        def render(predictions: np.ndarray):

            return [
                item_database[als.id_to_item_bmap[item_id]]
                # The minus in front of the prediction (ndarray) makes `np.argsort`
                # to do the arg-sorting in descending order. The hardcoded `10` should
                # be a parameter provided through the recommender object.
                for item_id in np.argsort(-predictions)[:10]
            ]

        return Predictor(predict_func=predict, render_func=render)


class AlternatingLeastSquares(Algorithm):
    """
    Alternating Least Squares algorithm. In the design, we assume that an instance of an
    algorithm is a process that is just waiting for data to run. And that process state
    be changed as it is being run. So one needs to instance another algorithm instance
    each time.

    This way of thinking makes the implementation easier than assuming that an algorithm
    instance's states should not change in terms of its extrinsic states (the intrinsic
    states of an algorithm are the hyperparameters), which will require us to expose the
    states change using another pattern and that seems more complex.
    """

    # The client code needs this in other to pull intrinsic params dynamically
    HYPER_PARAMETERS = [
        "hyper_lambda",
        "hyper_gamma",
        "hyper_tau",
        "hyper_n_epochs",
        "hyper_n_factors",
    ]

    PREDICTION_EPOCHS = 4

    AlternatingLeastSquaresError = type(
        "AlternatingLeastSquaresError", (Exception,), {}
    )

    def __init__(
        self,
        hyper_lambda: float = 0.1,
        hyper_gamma: float = 0.01,
        hyper_tau: float = 0.1,
        hyper_n_epochs: int = 10,
        hyper_n_factors: int = 10,
        user_factors: Optional[np.ndarray] = None,
        item_factors: Optional[np.ndarray] = None,
        user_biases: Optional[np.ndarray] = None,
        item_biases: Optional[np.ndarray] = None,
        feature_factors: Optional[np.ndarray] = None,
    ):

        assert (
            hyper_lambda
            and hyper_gamma
            and hyper_tau
            and hyper_n_factors
            # and hyper_n_epochs
        ), (
            # Serves as a message
            hyper_lambda,
            hyper_gamma,
            hyper_tau,
            hyper_n_factors,
            # hyper_n_epochs,
        )

        self.hyper_lambda = hyper_lambda
        self.hyper_gamma = hyper_gamma
        self.hyper_tau = hyper_tau
        self.hyper_n_epochs = hyper_n_epochs
        self.hyper_n_factors = hyper_n_factors

        self.user_factors = user_factors
        self.item_factors = item_factors
        self.user_biases = user_biases
        self.item_biases = item_biases
        # The feature factors
        self.feature_factors = feature_factors

        # This is useful, when the backend (i.e., the caller) wants
        # to resume the training from certain epoch instead of starting
        # completing from the beginning. In such setting, the initial
        self._initial_hyper_n_epochs = hyper_n_epochs

        self._state_resumed = False
        self._include_features = (False,)

        self._epochs_loss_train = []
        self._epochs_loss_test = []
        self._epochs_rmse_train = []
        self._epochs_rmse_test = []

        # The two following attributes rely on the data (indexed data) that will be
        # passed to the `run` method. And they are used to get a user's id or an
        # item's id if we know the user or the item or inversely. This design makes
        # realizing that it is better to pass the data while instantiation this class.
        # We'll leave it as improvement to be studied.

        self.id_to_user_bmap: Optional[SerialBidirectionalMapper] = None
        self.id_to_item_bmap: Optional[SerialBidirectionalMapper] = None
        # This is useful for the feature learning
        self.item_database: Optional[dict] = None

    @staticmethod
    def _validate_dimension_equality(*arrays):
        """
        Validates that all provided arrays or sequences are of the same
        type and have matching dimensions.

        Parameters:
            arrays: A variable number of arrays or sequences to compare.

        Raises:
            TypeError: If the arrays have mismatched types, shapes, or lengths.
        """

        # This method is ugly :) , but it is okay for now.
        type_ = type(arrays[0])

        if not all(isinstance(arr, (type_, NoneType)) for arr in arrays):  # Skip None
            raise TypeError("All arrays or sequences must be of the same type.")

        if type_ == np.ndarray and (
            not all(arr.shape == arrays[0].shape for arr in arrays if arr is not None)
        ):
            raise TypeError(
                "Arrays have mismatched shapes. Ensure all inputs are ndarrays with the same shape"
            )

        if type_ == list and (
            not all(len(arr) == len(arrays[0]) for arr in arrays if arr is not None)
        ):
            raise TypeError(
                "Lists have mismatched lengths. Ensure all inputs are lists with the same length."
            )

    def _validate_epochs_losses_and_rmse(
        self, loss_train, loss_test, rmse_train, rmse_test
    ):
        """
        Validates that all epoch-related lists (loss and RMSE) have the same length.
        """
        try:
            self._validate_dimension_equality(
                loss_train, loss_test, rmse_train, rmse_test
            )
        except TypeError as exc:
            raise self.AlternatingLeastSquaresError(
                f"Expected loss_train, loss_test, rmse_train, and rmse_test to have the same length, "
                f"but got lengths {len(loss_train)}, {len(loss_test)}, {len(rmse_train)}, and {len(rmse_test)}."
            ) from exc

    def _load_state(self, state: AlternatingLeastSquaresState):
        """
        Internal method to update the state of the algorithm.
        This is not exposed to client code to ensure encapsulation.
        """
        if (
            not (feature_factor := getattr(state, "feature_factors", None))
            and self._include_features
        ):
            logger.error(
                "State that is being loaded does not have `feature_factors`, so cannot run {self.__class__.__name__} "
                "algorithm with feature included, exiting..."
            )

        self.__init__(
            hyper_lambda=state.hyper_lambda,
            hyper_gamma=state.hyper_gamma,
            hyper_tau=state.hyper_tau,
            hyper_n_epochs=state.hyper_n_epochs,
            hyper_n_factors=state.hyper_n_factors,
            user_factors=state.user_factors,
            item_factors=state.item_factors,
            user_biases=state.user_biases,
            item_biases=state.item_biases,
            # To support old models and not crash
            feature_factors=feature_factor,
        )

        self._epochs_loss_train = state.loss_train or []
        self._epochs_loss_test = state.loss_test or []
        self._epochs_rmse_train = state.rmse_train or []
        self._epochs_rmse_test = state.rmse_test or []

        # Helps to know whether the state has been resumed or not
        self._state_resumed = True

    @property
    def state(self) -> AlgorithmState:

        return AlternatingLeastSquaresState(
            {
                # Non changing states (intrinsic)
                "hyper_lambda": self.hyper_lambda,
                "hyper_tau": self.hyper_tau,
                "hyper_gamma": self.hyper_gamma,
                "hyper_n_epochs": self.hyper_n_epochs,
                "hyper_n_factors": self.hyper_n_factors,
                # The states that change (extrinsic)
                "user_factors": self.user_factors,
                "item_factors": self.item_factors,
                "user_biases": self.user_biases,
                "item_biases": self.item_biases,
                "feature_factors": self.feature_factors,
                "loss_train": self._epochs_loss_train,
                "loss_test": self._epochs_loss_test,
                "rmse_train": self._epochs_rmse_train,
                "rmse_test": self._epochs_rmse_test,
            }
        )

    @cached_property
    def _n_feature(self):
        """
        The number of item features modeled
        """
        return (
            len(self.item_database[self.id_to_item_bmap[0]]["features_hot_encoded"])
            if self._include_features
            else 0
        )

    def _validate_state(self, state):
        self._validate_epochs_losses_and_rmse(
            state.loss_train, state.loss_test, state.rmse_train, state.rmse_test
        )

    def _finalize_factors_and_biases_initialization(
        self, data_by_user_id__train, data_by_item_id__train
    ):
        """
        Initialize factors and biases based on the information provided while
        creating an instance of the algorithm. Like either learn factors and biases
        or initialize using the respective distributions they are coming from.
        """

        users_count = len(data_by_user_id__train)
        items_count = len(data_by_item_id__train)

        if self.feature_factors is None:
            # TODO: Beta should be an hyper parameter
            self.feature_factors = self._get_factor_sample(
                size=(self._n_feature, self.hyper_n_factors),
                scale=settings.als.HYPER_BETA,
            )

        # If we know user factors and user biases but item factors and biases are not known,
        # we can learn them using user factors and user biases that we know. And inversely,
        # if we know item factors and item biases but user factors and biases are unknown, we
        # can learn them too. But we're going to provide a simplified implementation of it now.

        if (
            self.user_factors is None
            and self.user_biases is None
            and self.item_factors is None
            and self.item_biases is None
        ):
            logger.info(
                "Initializing user and item's factors and biases, as none of them is provided."
            )
            self.user_factors = self._get_factor_sample(
                size=(users_count, self.hyper_n_factors)
            )
            self.item_factors = self._get_factor_sample(
                size=(items_count, self.hyper_n_factors)
            )
            self.user_biases = self._get_bias_sample(users_count)
            self.item_biases = self._get_bias_sample(items_count)

        elif self.item_factors is None and self.item_biases is None:
            # Initialize item factors and biases and then update the factors and biases via learning
            logger.info(
                "Learning item factors and biases using the provided `user_factors` and `user_biases`..."
            )
            self.item_factors = self._get_factor_sample(
                size=(items_count, self.hyper_n_factors)
            )
            self.item_biases = self._get_bias_sample(items_count)
            for item_id in data_by_item_id__train:
                self.update_item_bias_and_factor(
                    item_id, data_by_item_id__train[item_id]
                )
        elif self.user_factors is None and self.user_biases is None:
            # Initialize user factors and biases and then update the factors and biases via learning
            logger.info(
                "Learning user factors and biases using the provided `item_factors` and `item_biases`..."
            )

            self.user_factors = self._get_factor_sample(
                size=(users_count, self.hyper_n_factors)
            )

            self.user_biases = self._get_bias_sample(size=users_count)
            for user_id in data_by_user_id__train:
                self.update_user_bias_and_factor(
                    user_id, data_by_user_id__train[user_id]
                )
        else:
            # We have all the factors and biases defined, so nothing to do
            logger.info(
                "All factors and biases are already provided, so no initialization is needed."
            )

    def _learn_bias_and_factor(
        self,
        target: LearningTargetEnum,
        target_id: Optional[int] = None,
        target_factor: np.ndarray = None,
        ratings_data: Optional[list] = None,
    ):
        """
        Learn or compute user or item (target) related bias and factor based on the
        provided ratings data and the actual state of the item biases and factors.

        NOTE: In the code, target is either "user" or "item". And we use "target" and
        "other_target" to designate both of them.

        IMPORTANT: `target` does not designate a specific an instance of "user" or "item".

        TODO: This ambiguous can be avoided by using the using `target_instance` for instances
        """

        _targets = LearningTargetEnum.targets()

        _mapping = {
            LearningTargetEnum.USER: (
                self.user_factors,
                self.user_biases,
                self.id_to_user_bmap,
            ),
            LearningTargetEnum.ITEM: (
                self.item_factors,
                self.item_biases,
                self.id_to_item_bmap,
            ),
        }

        _index = _targets.index(target)

        # If the old factor has not been passed, then try to access it
        if target_factor is None:
            # Get the target factors to attempt to retrieve the old factor from
            # which we want to learn the bias and them the updated version of the
            # factor.

            target_factors, _, _id_to_target_bmap = _mapping[_targets[_index]]
            # The old factor that we want to use in other to learn a new one
            factor = (
                target_factors[target_id]
                if target_id
                else self._get_factor_sample(size=self.hyper_n_factors)
            )
        else:
            factor = target_factor

        (other_target_factors, other_target_biases, _id_to_other_target_bmap) = (
            _mapping[_targets[1 - _index]]
        )

        bias = 0

        ratings_count = 0

        _A = np.zeros((self.hyper_n_factors, self.hyper_n_factors))
        _B = np.zeros(self.hyper_n_factors)

        for data in ratings_data:
            other_target, rating = (
                # Access the other target
                data[0],
                # Access the rating
                data[1],
            )
            other_target_id = _id_to_other_target_bmap.inverse[other_target]

            bias += (
                rating
                - other_target_biases[other_target_id]
                - np.dot(factor, other_target_factors[other_target_id])
            )
            ratings_count += 1

        bias = (self.hyper_lambda * bias) / (
            self.hyper_lambda * ratings_count + self.hyper_gamma
        )

        for data in ratings_data:
            other_target, rating = (
                data[0],
                data[1],
            )
            other_target_id = _id_to_other_target_bmap.inverse[other_target]

            _A += np.outer(
                other_target_factors[other_target_id],
                other_target_factors[other_target_id],
            )
            _B += (
                rating - bias - other_target_biases[other_target_id]
            ) * other_target_factors[other_target_id]

        # Adding of the feature factor related term it should be done
        if target == LearningTargetEnum.ITEM and self._include_features:
            _B += self._get_accumulated_scaled_feature_factor(target_id)

        factor = np.linalg.solve(
            self.hyper_lambda * _A + self.hyper_tau * np.eye(self.hyper_n_factors),
            self.hyper_lambda * _B,
        )

        return factor, bias

    def learn_user_bias_and_factor(
        self,
        user_id: Optional[int] = None,
        user_factor: np.ndarray = None,
        user_ratings_data: Optional[list] = None,
    ):
        """
        Learn or compute the given user_id related bias and factor based on the
        provided ratings data and the actual state of the item biases and factors.
        """

        return self._learn_bias_and_factor(
            target=LearningTargetEnum.USER,
            target_id=user_id,
            target_factor=user_factor,
            ratings_data=user_ratings_data,
        )

    def learn_item_bias_and_factor(
        self,
        item_id: Optional[int] = None,
        item_factor: Optional[np.ndarray] = None,
        item_ratings_data: Optional[list] = None,
    ):
        """
        Learn or compute the given item_id related bias and factor based on the
        provided ratings data and the actual state of the user biases and factors.
        """

        return self._learn_bias_and_factor(
            target=LearningTargetEnum.ITEM,
            target_id=item_id,
            target_factor=item_factor,
            ratings_data=item_ratings_data,
        )

    def learn_feature_factor(
        self,
        feature_id: Optional[int] = None,
        feature_factor=None,
    ) -> np.ndarray:
        """
        Learn or compute the given feature_id related factor using the item vectors
        """

        # Access _als_cache
        global _als_cache

        if not (item_features_counts := _als_cache.get("item_features_counts")):
            item_features_counts = np.zeros(shape=len(self.id_to_item_bmap))

            for i, item in enumerate(self.id_to_item_bmap.inverse):
                item_features_counts[i] = self.item_database[item]["features_count"]

            # DOCME: Here we will do an engineering hack to avoid zero division error.
            #   For the items that have no feature, we will set their feature count to
            #   infinity to make kinda nullify the factor `1 / np.sqrt(feature_count)`.
            #   This idea makes sens because the feature term does not make sens for item
            #   without features, which means that we should not include the feature extraction
            #   term for those items.

            item_features_counts[item_features_counts == 0] = sys.maxsize  #

            # Cache the result for the next time
            _als_cache["features_counts"] = item_features_counts

        # Accumulate the weighted feature factor for all items
        accumulated_weighted_feature_factor = np.zeros(shape=(1, self.hyper_n_factors))
        for item_id, item in enumerate(self.id_to_item_bmap.inverse):
            item_features_hot_encoded = self.item_database[item]["features_hot_encoded"]
            mask = np.array(item_features_hot_encoded, dtype=bool) & (
                np.arange(len(item_features_hot_encoded)) != item_id
            )
            accumulated_weighted_feature_factor += (
                self.feature_factors[mask].sum(axis=0) / item_features_counts[item_id]
            )

        numerator = (
            np.sum(
                np.multiply(
                    1 / np.sqrt(item_features_counts.reshape(-1, 1)), self.item_factors
                ),
                axis=0,
            )
            - accumulated_weighted_feature_factor
        )

        denominator = 1 + (1 / np.sqrt(item_features_counts.reshape(-1, 1))).sum()

        return numerator / denominator

    def _get_factor_sample(
        self, size, loc: Optional[float] = None, scale: Optional[float] = None
    ) -> np.ndarray:
        """
        Returns a factor sample using a normal distribution loc=`0` as
        mean and scale=`1 / np.sqrt(self.hyper_n_factors)` as a scale.
        """
        loc = 0.0 if loc is None else loc
        scale = 1 / np.sqrt(self.hyper_n_factors) if scale is None else scale
        return np.random.normal(
            loc=loc,
            scale=scale,
            size=size,
        )

    @staticmethod
    def _get_bias_sample(size) -> np.ndarray:
        """
        Returns a bias sample initialized to zeros with the given size.
        """
        return np.zeros(size)

    @staticmethod
    def _compute_rmse(
        accumulated_squared_residual: float, residuals_count: int
    ) -> float:
        """
        Returns the Root Mean Squared Error
        """
        return np.sqrt(accumulated_squared_residual / residuals_count)

    def _compute_loss(self, accumulated_squared_residual: float) -> float:
        return (
            (-1 / 2 * self.hyper_lambda * accumulated_squared_residual)
            - self.hyper_tau / 2 * self._get_accumulated_factors_product()
            - self.hyper_gamma * self._get_accumulated_squared_biases()
        )

    def _get_accumulated_scaled_feature_factor(self, item_id):
        """
        Calculate the accumulated scaled feature factor for a given item_id.
        """

        item = self.id_to_item_bmap[item_id]
        features_hot_encoded = self.item_database[item]["features_hot_encoded"]
        mask = np.array(features_hot_encoded, dtype=bool)

        return (
            self.hyper_tau
            * self.feature_factors[mask].sum(axis=0)
            / self.item_database[item]["features_count"]
        )

    def _get_accumulated_squared_residual_and_residuals_count(
        self, data_by_user_id: SerialUnidirectionalMapper
    ) -> tuple[float, int]:
        """
        Compute the accumulated squared residuals and their count for the given data.
        """
        accumulated_squared_residuals = 0
        residuals_count = 0
        for user_id in data_by_user_id:
            for data in data_by_user_id[user_id]:
                item, user_item_rating = data
                item_id = self.id_to_item_bmap.inverse[item]
                accumulated_squared_residuals += (
                    user_item_rating
                    - (
                        self.user_biases[user_id]
                        + self.item_biases[item_id]
                        + np.dot(self.user_factors[user_id], self.item_factors[item_id])
                    )
                ) ** 2

                residuals_count += 1
        return accumulated_squared_residuals, residuals_count

    def _get_accumulated_squared_biases(self):
        # This way of computing the accumulated biases norm is faster
        return np.square(self.user_biases).sum() + np.square(self.item_biases).sum()

    def _get_accumulated_factors_product(self):
        """
        Compute the Frobenious squared norm for "user", "item", "feature"
        # https://mathworld.wolfram.com/FrobeniusNorm.html#:~:text=The%20Frobenius%20norm%2C%20sometimes%20also,considered%20as%20a%20vector%20norm.
        """
        accumulated_squared_user_factor = np.square(self.user_factors).sum()
        accumulated_squared_item_factor = np.square(
            [
                self.item_factors[item_id]
                - (
                    self._get_accumulated_scaled_feature_factor(item_id)
                    if self._include_features
                    else 0
                )
                for item_id in range(len(self.item_factors))
            ]
        ).sum()
        accumulated_squared_feature_factor = (
            np.square(self.feature_factors).sum() if self._include_features else 0
        )
        return (
            accumulated_squared_user_factor
            + accumulated_squared_item_factor
            + accumulated_squared_feature_factor
        )

    def update_user_bias_and_factor(self, user_id, user_ratings_data: list):
        """
        Side effect method that updates the given user's bias and latent factor
        """
        user_factor, user_bias = self.learn_user_bias_and_factor(
            user_id=user_id, user_ratings_data=user_ratings_data
        )
        self.user_biases[user_id] = user_bias
        self.user_factors[user_id] = user_factor

    def update_item_bias_and_factor(self, item_id, item_ratings_data: list):
        """
        Side effect method that updates the given item's bias and latent factor
        """
        item_factor, item_bias = self.learn_item_bias_and_factor(
            item_id=item_id, item_ratings_data=item_ratings_data
        )
        self.item_biases[item_id] = item_bias
        self.item_factors[item_id] = item_factor

    def update_feature_factor(self, feature_id):
        feature_factor = self.learn_feature_factor(feature_id=feature_id)
        self.feature_factors[feature_id] = feature_factor

    def run(
        self,
        data: IndexedDatasetWrapper,
        initial_state: AlternatingLeastSquaresState = None,
        item_database: dict = None,
        include_features: bool = False,
    ):
        """
        Runs the algorithm on the indexed data, `IndexedDatasetWrapper`.
        """

        assert isinstance(
            data, IndexedDatasetWrapper
        ), "The provided `indexed_data` must be an instance of `IndexedDatasetWrapper`."

        assert (
            include_features and item_database
        ) or not include_features, "When `include_features` is set to `True` the `item_database` should be a defined dict"

        data_by_user_id__train = data.data_by_user_id__train
        data_by_item_id__train = data.data_by_item_id__train

        # The validation data, just to compute the loss for the validation data too.
        # But here we're not distinguishing the validation set from the training's one.
        data_by_user_id__test = data.data_by_user_id__test

        # TODO: Needs doc
        # data_by_item_id__test = indexed_data.data_by_item_id__test

        # We want to run the algorithm in resume mode from the backend
        if initial_state:
            state = (
                AlternatingLeastSquaresState(initial_state)
                if isinstance(initial_state, dict)
                else initial_state
            )

            # Run some validation rules
            self._validate_state(state)

            self._load_state(state)

        self.id_to_user_bmap = data.id_to_user_bmap
        self.id_to_item_bmap = data.id_to_item_bmap

        # About the feature integration
        self.item_database = item_database
        self._include_features = include_features

        # Check if factors and biases are set or learn them if applicable
        self._finalize_factors_and_biases_initialization(
            data_by_user_id__train, data_by_item_id__train
        )

        logger.info(
            f"About to start training with the `include_features` parameter set to {self._include_features}."
        )

        n_epochs = self.hyper_n_epochs

        # When we resume, we only want to train for the rest of the epochs
        if self._state_resumed and (
            (n_epochs := self._initial_hyper_n_epochs - self.hyper_n_epochs) <= 0
        ):
            logger.error(
                f"Cannot train the model more because hyperparameter 'hyper_n_epochs' ({self.hyper_n_epochs}) is already "
                f"greater or equal to the final number of epochs wanted which is {self._initial_hyper_n_epochs}. "
                "Please check the value of 'hyper_n_epochs' and adjust accordingly. Exiting..."
            )
            return

        logger.info(
            f"Epochs count to train for {n_epochs}, entering the training loop now..."
        )

        # Get the len of the first hot encoded features array
        n_features = (
            len(self.item_database[self.id_to_item_bmap[0]]["features_hot_encoded"])
            if self._include_features
            else 0
        )

        for epoch in tqdm(range(n_epochs), desc="Epochs", unit="epoch"):

            for user_id in data_by_user_id__train:
                self.update_user_bias_and_factor(
                    user_id, data_by_user_id__train[user_id]
                )

            for item_id in data_by_item_id__train:
                self.update_item_bias_and_factor(
                    item_id, data_by_item_id__train[item_id]
                )

            for feature_id in range(n_features):
                self.update_feature_factor(feature_id)

            # We've got a new model from the current epoch, so time to compute
            # the metrics for both the training and test set.

            accumulated_squared_residual_train, residuals_count_train = (
                self._get_accumulated_squared_residual_and_residuals_count(
                    data_by_user_id__train
                )
            )

            accumulated_squared_residual_test, residuals_count_test = (
                self._get_accumulated_squared_residual_and_residuals_count(
                    data_by_user_id__test
                )
            )

            loss_train = self._compute_loss(accumulated_squared_residual_train)
            loss_test = self._compute_loss(accumulated_squared_residual_test)

            # We are assuming that this code neither runs in production .i.e is not end client code.
            # If that assumption changes, the following line should be replaced by a proper raise
            # of an exception.
            assert residuals_count_train and residuals_count_test, (
                "None of `residuals_count_train` or `residuals_count_test` should be zero but "
                f"got {residuals_count_train} and {residuals_count_test} respectively for them. "
                "It happens because the data for which that count comes from is empty."
            )

            rmse_train = self._compute_rmse(
                accumulated_squared_residual_train, residuals_count_train
            )
            rmse_test = self._compute_rmse(
                accumulated_squared_residual_test, residuals_count_test
            )

            self._epochs_loss_train.append(loss_train)
            self._epochs_loss_test.append(loss_test)
            self._epochs_rmse_train.append(rmse_train)
            self._epochs_rmse_test.append(rmse_test)

            # Print the information for the current epoch
            tqdm.write(
                f"Epoch {epoch + 1}/{self.hyper_n_epochs} "
                f"Loss (Train/Test) : {loss_train:.4f} / {loss_test:.4f}, "
                f"RMSE (Train/Test) : {rmse_train:.4f} / {rmse_test:.4f}"
            )

        logger.info(
            f"Successfully run {self.__class__.__name__} algorithm running till the end"
        )
        logger.debug(
            f"Cleaning the {self.__class__.__name__} algorithm self maintained cache, and exiting..."
        )

        _clear_als_cache()
