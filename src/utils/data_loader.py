import logging
import math
import random
import numpy as np
from csv import DictReader
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Iterable, Callable, Sequence, Optional

from src.utils.mapper import SerialBidirectionalMapper, SerialUnidirectionalMapper
from src.utils.constants import NOT_PROVIDED, NOT_DEFINED

logger = logging.getLogger(__name__)

sample_from_bernoulli = partial(np.random.binomial, n=1)


@dataclass
class LoadedDataWrapper:
    data_by_user_id__train: SerialUnidirectionalMapper
    data_by_item_id__train: SerialUnidirectionalMapper
    data_by_user_id__test: SerialUnidirectionalMapper
    data_by_item_id__test: SerialUnidirectionalMapper
    # Mapping data
    id_to_user_bmap: SerialBidirectionalMapper
    id_to_item_bmap: SerialBidirectionalMapper


class AbstractDataLoader(ABC):

    @abstractmethod
    def load(self):
        pass


class DataLoader(AbstractDataLoader):
    """
    We want to load the data as an optimized form of sparse matrices
    """

    LIMIT_TO_LOAD_IN_MEMORY = 1_000_000_000_000

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
        self._limit = limit or self.LIMIT_TO_LOAD_IN_MEMORY

    def _construct_data(self, data, items):  # noqa
        """
        Add structure to data
        """
        return (
            self._data_constructor(data, items)
            if self._data_constructor is not None
            else data
        )

    def load(self, approximate_train_ratio: float = 1) -> LoadedDataWrapper:

        assert (
            0 <= approximate_train_ratio <= 1
        ), f"Expected `train_ratio` to be in [0, 1] but got {approximate_train_ratio}"

        logger.warning(
            "The current implementation does not split the data into train and test sets exactly with the provided ration. "
            "We use the provided ratio as a probability for a Bernoulli distribution to know either the data point should "
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

                    # Whether the current entry should be pushed into the training
                    # set or not.
                    belongs_to_test_split = NOT_PROVIDED

                    # Check if the user and the item already exists.

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

                    if belongs_to_test_split is NOT_DEFINED:
                        # Search for where the user is and add the data there
                        if data_by_user_id__train[user_id]:
                            data_by_user_id__train.add(data=data, key=user_id)
                            data_by_item_id__train.add(data=data, key=item_id)
                        else:
                            data_by_user_id__test.add(data=data, key=user_id)
                            data_by_item_id__test.add(data=data, key=item_id)
                    elif belongs_to_test_split is True:
                        data_by_user_id__test.add(data=data, key=user_id)
                        data_by_item_id__test.add(data=data, key=item_id)
                    else:
                        data_by_user_id__train.add(data=data, key=user_id)
                        data_by_item_id__train.add(data=data, key=item_id)

                    if self._verbose:
                        logger.info(
                            f"Load the line {indexed_count} of {self._file_path} successfully"
                        )

                    indexed_count += 1
                    if indexed_count == self._limit:
                        logger.warning(
                            f"Limit of entries (.i.e {self._limit}) to load has been reached. Exiting without loading the rest... "
                        )
                        break
        except (FileNotFoundError, KeyError) as exc:
            logger.error(
                f"Cannot load data from the given `file_path` .i.e {self._file_path}. Attempt failed with exception {exc}"
            )
            raise self.DataLoadingError() from exc

        logger.info(
            f"Successfully loaded {indexed_count} lines from {self._file_path} successfully"
        )
        return LoadedDataWrapper(
            data_by_user_id__train=data_by_user_id__train,
            data_by_item_id__train=data_by_item_id__train,
            data_by_user_id__test=data_by_user_id__test,
            data_by_item_id__test=data_by_item_id__test,
            id_to_user_bmap=id_to_user_bmap,
            id_to_item_bmap=id_to_item_bmap,
        )

    def test_train_split(self, ratio: float):
        indexed_data = None  # TODO: FIX THIS
        data_by_user_id__train = SerialUnidirectionalMapper()  # noqa
        data_by_item_id__train = SerialUnidirectionalMapper()  # noqa
        data_by_user_id__test = SerialUnidirectionalMapper()  # noqa
        data_by_item_id__test = SerialUnidirectionalMapper()  # noqa

        visited_user_ids = set()
        visited_item_ids = set()

        dataset_indices_based_representation = [
            (user_id, user_item_id)
            for user_id in range(self.users_count)
            for user_item_id in range(self.user_id_degree_map[user_id])
        ]

        test_dataset_indices_based_representation = set(
            random.sample(
                dataset_indices_based_representation,
                math.floor(len(dataset_indices_based_representation) * ratio),
            )
        )

        for user_id, user_item_id in dataset_indices_based_representation:
            # Get the item_id using the `user_item_id`
            data = indexed_data.data_by_user_id[user_id][user_item_id]
            rating, item = data[self._rating_header], data[self._item_header]
            item_id = indexed_data.id_to_item_bmap.inverse[item]

            if (
                user_id,
                user_item_id,
            ) in test_dataset_indices_based_representation:

                # We are having this user for the first time
                if user_id not in visited_user_ids:
                    # Add the user and its first rating data
                    data_by_user_id__test.add(
                        indexed_data.data_by_user_id[user_id][user_item_id]
                    )
                    # Add the user_id but with an empty array; this helps to ensure that
                    # the test set and the training set will have the same dimension
                    data_by_user_id__train.add(SerialUnidirectionalMapper.NOTHING)
                    visited_user_ids.add(user_id)
                else:
                    # The user has already been added
                    data_by_user_id__test.add(
                        indexed_data.data_by_user_id[user_id][user_item_id], key=user_id
                    )

                # DEAL WITH ITEMS

                if item_id not in visited_item_ids:
                    data_by_item_id__test.add(
                        {
                            self._user_header: indexed_data.id_to_user_bmap[user_id],
                            self._rating_header: rating,
                        }
                    )
                    data_by_item_id__train.add(SerialUnidirectionalMapper.NOTHING)
                    visited_item_ids.add(item_id)
                else:
                    # The item has already been added both in the training and the test dataset
                    data_by_item_id__test.add(
                        {
                            self._user_header: indexed_data.id_to_user_bmap[user_id],
                            self._rating_header: rating,
                        },
                        key=item_id,
                    )
            else:
                # We've got an occurrence that should be used as a training dataset
                if user_id not in visited_user_ids:

                    data_by_user_id__train.add(
                        indexed_data.data_by_user_id[user_id][user_item_id]
                    )
                    # Add the user entry, but with an empty array
                    data_by_user_id__test.add(SerialUnidirectionalMapper.NOTHING)
                    visited_user_ids.add(user_id)
                else:
                    data_by_user_id__train.add(
                        indexed_data.data_by_user_id[user_id][user_item_id], key=user_id
                    )

                # DEAL WITH ITEMS

                if item_id not in visited_item_ids:
                    data_by_item_id__test.add(SerialUnidirectionalMapper.NOTHING)

                    data_by_item_id__train.add(
                        {
                            self._user_header: indexed_data.id_to_user_bmap[user_id],
                            self._rating_header: rating,
                        }
                    )

                    visited_item_ids.add(item_id)

                else:
                    # The item has already been added both in the training and the testing dataset
                    data_by_item_id__train.add(
                        {
                            self._user_header: indexed_data.id_to_user_bmap[user_id],
                            self._rating_header: rating,
                        },
                        key=item_id,
                    )

        return (data_by_user_id__train, data_by_item_id__train), (
            data_by_user_id__test,
            data_by_item_id__test,
        )