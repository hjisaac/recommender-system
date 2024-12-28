import box
import enum
import logging
import numpy as np
from typing import Optional, Literal
from collections import defaultdict

from src.algorithms.core import Algorithm
from src.utils.dataset_indexer import IndexedDatasetWrapper
from src.utils.state_manager import AlgorithmState
from src.utils.constants import LEARNING_TARGETS
from src.utils.serial_mapper import SerialUnidirectionalMapper

logger = logging.getLogger(__name__)


class AlternatingLeastSquares(Algorithm):

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
    ):

        assert (
            hyper_lambda
            and hyper_gamma
            and hyper_tau
            and hyper_n_factors
            and hyper_n_epochs
        ), (
            # Serves as message
            hyper_lambda,
            hyper_gamma,
            hyper_tau,
            hyper_n_factors,
            hyper_n_epochs,
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

        # The two following methods rely on the data (indexed data) that will be
        # passed to the `fit` method. And they are used to get a user's id or an
        # item's id if we know the user or the item.
        self.__get_user_id: Optional[defaultdict] = None
        self.__get_item_id: Optional[defaultdict] = None

    @property
    def state(self) -> AlgorithmState:

        return AlgorithmState(
            {
                "hyper_lambda": self.hyper_lambda,
                "hyper_tau": self.hyper_tau,
                "hyper_gamma": self.hyper_gamma,
                "hyper_n_epochs": self.hyper_n_epochs,
                "hyper_n_factors": self.hyper_n_factors,
                "user_factors": self.user_factors,
                "item_factors": self.item_factors,
                "user_biases": self.user_biases,
                "item_biases": self.item_biases,
            }
        )

    def _initialize_factors_and_biases(
        self, data_by_user_id__train, data_by_item_id__train
    ):
        users_count = len(data_by_user_id__train)
        items_count = len(data_by_item_id__train)

        # If we know user factors and user biases but item factors and biases are not known,
        # we can learn them using user factors and  user biases that we know. And inversely,
        # if we know item factors and item biases but user factors and biases are unknown we
        # can learn them too.
        if not (
            (self.user_factors and self.user_biases)
            or (self.item_factors and self.item_biases)
        ):
            logger.info(
                "Initializing user and item factors and biases, as none of them is provided."
            )
            self.user_factors = self._get_factor_sample(
                size=(users_count, self.hyper_n_factors)
            )
            self.item_factors = self._get_factor_sample(
                size=(items_count, self.hyper_n_factors)
            )
            self.user_biases = self._get_bias_sample(users_count)
            self.item_biases = self._get_bias_sample(items_count)

        elif self.user_factors and self.user_biases:
            # Initialize item factors and biases to and then update the factors and biases via learning
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
        elif self.item_factors and self.item_biases:
            # Initialize user factors and biases to zeros then update the factors and biases via learning
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
                "All factors and biases are already provided. No initialization is required."
            )

    def _learn_bias_and_factor(
        self,
        target: Literal["user" | "item"],
        id_=None,
        ratings_data: Optional[list] = None,
    ):
        """
        Learn or compute user or item (target) related bias and factor based on the
        provided ratings data and the actual state of the item biases and factors.

        NOTE: In the code, target is either "user" or "item". And we use "target" and
        "other_target" to designate both of them.
        """

        mapping_ = {
            "user": (self.user_factors, self.user_biases, self.__get_user_id),
            "item": (self.item_factors, self.item_biases, self.__get_item_id),
        }

        target_index = LEARNING_TARGETS.index[target]

        assert target_index, target_index

        # Get the target factors to attempt to retrieve the old factor from
        # which we want to learn the bias and them the updated version of the
        # factor.
        target_factors, _, __ = mapping_[LEARNING_TARGETS[target_index]]
        (other_target_factors, other_target_biases, _get_other_target_id) = mapping_[
            LEARNING_TARGETS[1 - target_index]
        ]

        bias = 0
        # The old factor that we want to use in other to learn a new one
        factor = (
            target_factors[id_]
            if id_
            else self._get_factor_sample(size=self.hyper_n_factors)
        )
        ratings_count = 0
        _A = np.zeros((self.hyper_n_factors, self.hyper_n_factors))
        _B = np.zeros(self.hyper_n_factors)

        for data in ratings_data:
            other_target, rating = data["movieId"], data["rating"]
            rating = float(rating)
            other_target_id = _get_other_target_id(target)

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
            other_target, rating = data["movieId"], data["rating"]
            rating = float(rating)
            other_target_id = _get_other_target_id(other_target)
            _A += np.outer(
                other_target_factors[other_target_id],
                other_target_factors[other_target_id],
            )
            _B += (
                rating - bias - other_target_biases[other_target_id]
            ) * other_target_factors[other_target_id]

        factor = np.linalg.solve(
            self.hyper_lambda * _A + self.hyper_tau * np.eye(self.hyper_n_factors),
            self.hyper_lambda * _B,
        )

        return factor, bias

    def learn_user_bias_and_factor(
        self, user_id=None, user_ratings_data: Optional[list] = None
    ):
        """
        Learn or compute the given user_id related bias and factor based on the
        provided ratings data and the actual state of the item biases and factors.
        """
        user_bias = 0
        user_factor = (
            self.user_factors[user_id]
            if user_id
            else self._get_factor_sample(size=self.hyper_n_factors)
        )
        ratings_count = 0
        _A = np.zeros((self.hyper_n_factors, self.hyper_n_factors))
        _B = np.zeros(self.hyper_n_factors)

        for data in user_ratings_data:
            item, user_item_rating = data["movieId"], data["rating"]
            user_item_rating = float(user_item_rating)
            item_id = self.__get_item_id(item)

            user_bias += (
                user_item_rating
                - self.item_biases[item_id]
                - np.dot(user_factor, self.item_factors[item_id])
            )
            ratings_count += 1

        user_bias = (self.hyper_lambda * user_bias) / (
            self.hyper_lambda * ratings_count + self.hyper_gamma
        )

        for data in user_ratings_data:
            item, user_item_rating = data["movieId"], data["rating"]
            user_item_rating = float(user_item_rating)
            item_id = self.__get_item_id(item)
            _A += np.outer(self.item_factors[item_id], self.item_factors[item_id])
            _B += (
                user_item_rating - user_bias - self.item_biases[item_id]
            ) * self.item_factors[item_id]

        user_factor = np.linalg.solve(
            self.hyper_lambda * _A + self.hyper_tau * np.eye(self.hyper_n_factors),
            self.hyper_lambda * _B,
        )

        return user_factor, user_bias

    def _get_factor_sample(self, size) -> np.ndarray:
        """
        Returns a factor sample using a normal distribution 0 as
        mean and `1 / np.sqrt(self.hyper_n_factors)` as scale.
        """
        return np.random.normal(
            loc=0.0,
            scale=1 / np.sqrt(self.hyper_n_factors),
            size=size,
        )

    @staticmethod
    def _get_bias_sample(size) -> np.ndarray:  # noqa
        """
        Returns a bias sample initialized to zeros with the given size.
        """
        return np.zeros(size)

    @staticmethod
    def _compute_rmse(
        accumulated_squared_residual: float, residuals_count: int
    ) -> float:
        return np.sqrt(accumulated_squared_residual / residuals_count)

    def _compute_loss(self, accumulated_squared_residual: float) -> float:
        return (
            (-1 / 2 * self.hyper_lambda * accumulated_squared_residual)
            - self.hyper_tau / 2 * sum(self._get_accumulated_factors_product())
            - self.hyper_gamma * sum(self._get_accumulated_squared_biases())
        )

    def _get_accumulated_squared_residual_and_count(
        self, data_by_user_id: SerialUnidirectionalMapper
    ) -> tuple[float, int]:
        accumulated_squared_residuals = 0
        residuals_count = 0
        for user_id in data_by_user_id:
            for data in data_by_user_id[user_id]:
                item, user_item_rating = data["movieId"], data["rating"]
                user_item_rating = float(user_item_rating)
                item_id = self.__get_item_id(item)
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
        return sum(bias**2 for bias in self.user_biases), sum(
            bias**2 for bias in self.item_biases
        )

    def _get_accumulated_factors_product(self):
        return sum(np.dot(factor, factor) for factor in self.user_factors), sum(
            np.dot(factor, factor) for factor in self.item_factors
        )

    def learn_item_bias_and_factor(
        self, item_id=None, item_ratings_data: Optional[list] = None
    ):
        """
        Learn or compute the given item_id related bias and factor based on the
        provided ratings data and the actual state of the user biases and factors.
        """

        item_bias = 0
        item_factor = (
            self.item_factors[item_id]
            if item_id
            else self._get_factor_sample(size=self.hyper_n_factors)
        )
        ratings_count = 0
        _A = np.zeros((self.hyper_n_factors, self.hyper_n_factors))
        _B = np.zeros(self.hyper_n_factors)

        for data in item_ratings_data:
            user, item_user_rating = data["userId"], data["rating"]
            item_user_rating = float(item_user_rating)
            user_id = self.__get_user_id(user)
            item_bias += (
                item_user_rating
                - self.user_biases[user_id]
                - np.dot(self.user_factors[user_id], item_factor)
            )
            ratings_count += 1

        item_bias = (self.hyper_lambda * item_bias) / (
            self.hyper_lambda * ratings_count + self.hyper_gamma
        )

        for data in item_ratings_data:
            user, item_user_rating = data["userId"], data["rating"]
            item_user_rating = float(item_user_rating)
            user_id = self.__get_user_id(user)

            _A += np.outer(self.user_factors[user_id], self.user_factors[user_id])
            _B += (
                item_user_rating - self.user_biases[user_id] - item_bias
            ) * self.user_factors[user_id]

        item_factor = np.linalg.solve(
            self.hyper_lambda * _A + self.hyper_tau * np.eye(self.hyper_n_factors),
            self.hyper_lambda * _B,
        )

        return item_factor, item_bias

    def update_user_bias_and_factor(self, user_id, user_ratings_data: list):
        """
        Side effect method that updates the given user's bias and latent factor
        """
        user_factor, user_bias = self.learn_user_bias_and_factor(
            user_id, user_ratings_data
        )
        self.user_biases[user_id] = user_bias
        self.user_factors[user_id] = user_factor

    def update_item_bias_and_factor(self, item_id, item_ratings_data: list):
        """
        Side effect method that updates the given item's bias and latent factor
        """
        item_factor, item_bias = self.learn_item_bias_and_factor(
            item_id, item_ratings_data
        )

        self.item_biases[item_id] = item_bias
        self.item_factors[item_id] = item_factor

    def fit(self, indexed_data: IndexedDatasetWrapper):

        data_by_user_id__train = indexed_data.data_by_user_id__train
        data_by_item_id__train = indexed_data.data_by_item_id__train

        # The validation data, just to compute the loss for the validation data too.
        # But here we're not distinguishing the validation set from the training's one.
        data_by_user_id__test = indexed_data.data_by_user_id__test

        # data_by_item_id__test = indexed_data.data_by_item_id__test

        self._initialize_factors_and_biases(
            data_by_user_id__train, data_by_item_id__train
        )

        self.__get_user_id = lambda user: indexed_data.id_to_user_bmap.inverse[user]
        self.__get_item_id = lambda item: indexed_data.id_to_item_bmap.inverse[item]

        for epoch in range(self.hyper_n_epochs):

            for user_id in data_by_user_id__train:
                self.update_user_bias_and_factor(
                    user_id, data_by_user_id__train[user_id]
                )

            for item_id in data_by_item_id__train:
                self.update_item_bias_and_factor(
                    item_id, data_by_item_id__train[item_id]
                )

            # We've got a new model from the current epoch, so time to compute
            # the metrics for both the training and test set.

            accumulated_squared_residual_train, residuals_count_train = (
                self._get_accumulated_squared_residual_and_count(data_by_user_id__train)
            )

            accumulated_squared_residual_test, residuals_count_test = (
                self._get_accumulated_squared_residual_and_count(data_by_user_id__test)
            )

            loss_train = self._compute_loss(accumulated_squared_residual_train)
            loss_test = self._compute_loss(accumulated_squared_residual_test)

            rmse_train = self._compute_rmse(
                accumulated_squared_residual_train, residuals_count_train
            )
            rmse_test = self._compute_rmse(
                accumulated_squared_residual_test, residuals_count_test
            )

            self.epochs_loss_train.append(loss_train)
            self.epochs_loss_test.append(loss_test)
            self.epochs_rmse_train.append(rmse_train)
            self.epochs_rmse_test.append(rmse_test)
