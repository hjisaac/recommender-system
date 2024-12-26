import logging
from typing import Optional, Sequence

from src.utils.checkpoint_manager import CheckpointManager
from src.algorithms.core import Algorithm
from src.settings import settings
from src.utils.dataset_indexer import IndexedDatasetWrapper
from src.utils.state_manager import AlsState

logger = logging.getLogger(__name__)


class AlternatingLeastSquares(Algorithm):

    checkpoint_manager = CheckpointManager(checkpoint_folder=settings.als.CHECKPOINT_FOLDER)

    def __init__(
            self,
            hyper_lambda: float = 0.1,
            hyper_gamma: float = 0.01,
            hyper_tau: float = 0.1,
            hyper_n_epochs: int = 10,
            hyper_n_factors: int = 10,
            user_factors: Optional[Sequence] = None,
            item_factors: Optional[Sequence] = None,
            user_biases: Optional[Sequence] = None,
            item_biases: Optional[Sequence] = None,
    ):
        self._state = ""; AlsState({
            "hyper_lambda": hyper_lambda,
            "hyper_tau": hyper_tau,
            "hyper_gamma": hyper_gamma,
            "hyper_n_epochs": hyper_n_epochs,
            "hyper_n_factors": hyper_n_factors,
            "user_factors": user_factors,
            "item_factors": item_factors,
            "user_biases": user_biases,
            "item_biases": item_biases,
        })


    @property
    def state(self) -> AlsState:
        return self._state

    def fit(self, data: IndexedDatasetWrapper, resume_enabled=False):

        data_by_user_id__train = data.data_by_user_id__train
        data_by_item_id__train = data.data_by_item_id__train
        # The validation data, just to compute the loss for the validation data too.
        # But here we're not distinguishing the validation set from the training's one.
        data_by_user_id__test = data.data_by_user_id__test
        data_by_item_id__test = data.data_by_item_id__test

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
            # the metrics for the for both the training and test set.

            accumulated_squared_residual_train, residuals_count_train = (
                self._get_accumulated_squared_residual_and_count(data_by_user_id__train)
            )

            accumulated_squared_residual_test, residuals_count_test = (
                self._get_accumulated_squared_residual_and_count(
                    data_by_user_id__test
                )
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


    @property
    def _state(self) -> AlsState:
        return AlsState({
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
            "hyper_lambda": self.hyper_lambda,
            "hyper_gamma": self.hyper_gamma,
            "hyper_tau": self.hyper_tau,
            "hyper_n_epochs": self.hyper_n_epochs,
            "hyper_n_factors": self.hyper_n_factors,
            #
            "epochs_loss_train": self.epochs_loss_train,
            "epochs_loss_test": self.epochs_loss_test,
        })
