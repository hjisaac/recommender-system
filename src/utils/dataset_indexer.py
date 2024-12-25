import logging
import math
import random
import numpy as np
from csv import DictReader
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Iterable, Callable, Sequence, Optional

from src.utils.serial_mapper import (
    SerialBidirectionalMapper,
    SerialUnidirectionalMapper,
)
from src.utils.constants import NOT_PROVIDED, NOT_DEFINED

logger = logging.getLogger(__name__)

sample_from_bernoulli = partial(np.random.binomial, n=1)


@dataclass
class IndexedDatasetWrapper:
    data_by_user_id__train: SerialUnidirectionalMapper
    data_by_item_id__train: SerialUnidirectionalMapper
    data_by_user_id__test: SerialUnidirectionalMapper
    data_by_item_id__test: SerialUnidirectionalMapper
    # Mapping data
    id_to_user_bmap: SerialBidirectionalMapper
    id_to_item_bmap: SerialBidirectionalMapper


class AbstractDatasetIndexer(ABC):

    @abstractmethod
    def index(self):
        pass


class DatasetIndexer(AbstractDatasetIndexer):
    """
    We want to index the dataset as an optimized form of sparse matrices
    """

    LIMIT_TO_INDEX_IN_MEMORY = 1_000_000_000_000

    DataLoadingError = type("InvalidStateError", (Exception,), {})

    def __init__(
        self,
        *,
        file_path: str,
        user_header: str,
        item_header: str,
        rating_header: str,
        data_headers: Iterable = None,
        data_constructor: Callable = None,
        verbose=True,
        limit: int = None,
    ):
        self._user_header = user_header
        self._item_header = item_header
        self._rating_header = rating_header
        self._file_path = file_path
        self._data_headers = data_headers
        self._data_constructor = data_constructor
        self._verbose = verbose
        self._limit = limit or self.LIMIT_TO_INDEX_IN_MEMORY

    def _construct_data(self, data, items):  # noqa
        """
        Add structure to data
        """
        return (
            self._data_constructor(data, items)
            if self._data_constructor is not None
            else data
        )

    def index(self, approximate_train_ratio: float = 1) -> IndexedDatasetWrapper:

        assert (
            0 <= approximate_train_ratio <= 1
        ), f"Expected `train_ratio` to be in [0, 1] but got {approximate_train_ratio}"

        logger.warning(
            "The current implementation does not split the data into train and test sets exactly with the provided ratio. "
            "We use the provided ratio as a probability for a Bernoulli distribution to know whether the data point should "
            "be used as a training data or a test data."
        )

        indexed_count = 0

        # The two next bmap variables are needed to keep track of the bijective
        # mapping between each user or item and the id we've associated to them.
        id_to_item_bmap = SerialBidirectionalMapper()  # bijective mapping
        id_to_user_bmap = SerialBidirectionalMapper()

        # This two next variables are used to group data by item_id and user_id
        data_by_user_id__train = SerialUnidirectionalMapper()  # subjective mapping
        data_by_item_id__train = SerialUnidirectionalMapper()

        data_by_user_id__test = SerialUnidirectionalMapper()  # subjective mapping
        data_by_item_id__test = SerialUnidirectionalMapper()

        belongs_to_test_split: Optional[bool | object]
        try:
            with open(self._file_path, mode="r", newline="") as csvfile:
                for line_count, line in enumerate(DictReader(csvfile)):
                    user, item = line[self._user_header], line[self._item_header]

                    # Unlikely in this dataset but better have this guard
                    if not user or not item:
                        logger.warning(
                            f"Cannot process the line {line_count}, cause expects `user` and `item` to be defined but got {user} and {item} for them respectively, skipping..."
                        )
                        continue

                    user_id = id_to_user_bmap.inverse[user]
                    item_id = id_to_item_bmap.inverse[item]

                    # Whether the current entry should be pushed into the training set
                    # or not. Set to `NOT_PROVIDED` to express the fact that the user
                    # has already been seeing once and then already belongs to the test
                    # set or the training set, no need to sample from Bernoulli distro
                    # in that case.
                    belongs_to_test_split = NOT_PROVIDED

                    if user_id is None:
                        # This user is a new one, so add it
                        id_to_user_bmap.add(user)

                        # If the user is a new one, run a Bernoulli experiment to
                        # decide whether that new user's data should be used as a
                        # training data of test data.
                        belongs_to_test_split = not sample_from_bernoulli(
                            p=approximate_train_ratio
                        )

                    if item_id is None:
                        # This item is a new one, so add it
                        id_to_item_bmap.add(item)

                    # Add the data without the useless items for indexing
                    data = self._construct_data(line, self._data_headers)

                    # Like if the user have already been seen once
                    if belongs_to_test_split is NOT_DEFINED:
                        # Search for where the user is and add the data there
                        if data_by_user_id__train[user_id]:
                            data_by_user_id__train.add(data=data, key=user_id)
                            data_by_item_id__train.add(data=data, key=item_id)

                            if item_id is None:
                                data_by_item_id__test.add(
                                    SerialUnidirectionalMapper.NOTHING
                                )

                        else:
                            data_by_user_id__test.add(data=data, key=user_id)
                            data_by_item_id__test.add(data=data, key=item_id)

                            if item_id is None:
                                data_by_item_id__train.add(
                                    SerialUnidirectionalMapper.NOTHING
                                )

                    elif belongs_to_test_split is True:
                        data_by_user_id__test.add(data=data, key=user_id)
                        data_by_item_id__test.add(data=data, key=item_id)

                        # Add the user_id to the training set but without the data to
                        # keep the same dimension between the training and test set
                        data_by_user_id__train.add(SerialUnidirectionalMapper.NOTHING)
                        # But if the item_id already exists do nothing
                        if item_id is None:
                            data_by_item_id__train.add(
                                SerialUnidirectionalMapper.NOTHING
                            )

                    else:
                        data_by_user_id__train.add(data=data, key=user_id)
                        data_by_item_id__train.add(data=data, key=item_id)
                        # Add the user_id to the test set but without the data to
                        # keep the same dimension between the training and test set
                        data_by_user_id__test.add(SerialUnidirectionalMapper.NOTHING)

                        # But if the item_id already exists do nothing
                        if item_id is None:
                            data_by_item_id__test.add(
                                SerialUnidirectionalMapper.NOTHING
                            )

                    if self._verbose:
                        logger.info(
                            f"Load the line {indexed_count} of {self._file_path} successfully"
                        )

                    indexed_count += 1
                    if indexed_count == self._limit:
                        logger.warning(
                            f"Limit of lines (.i.e {self._limit}) to load has been reached. Exiting without loading the rest... "
                        )
                        break
        except (FileNotFoundError, KeyError) as exc:
            logger.error(
                f"Cannot load data from the given `file_path` .i.e {self._file_path}. Attempt failed with exception {exc}"
            )
            raise self.DataLoadingError() from exc

        logger.info(f"Successfully loaded {indexed_count} lines from {self._file_path}")

        return IndexedDatasetWrapper(
            data_by_user_id__train=data_by_user_id__train,
            data_by_item_id__train=data_by_item_id__train,
            data_by_user_id__test=data_by_user_id__test,
            data_by_item_id__test=data_by_item_id__test,
            id_to_user_bmap=id_to_user_bmap,
            id_to_item_bmap=id_to_item_bmap,
        )
