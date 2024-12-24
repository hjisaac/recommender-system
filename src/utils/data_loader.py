import logging
import math
import random
from csv import DictReader
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Sequence, Optional

from ..utils.mapper import SerialBidirectionalMapper, SerialUnidirectionalMapper

logger = logging.getLogger(__name__)


class AbstractDataLoader(ABC):

    @abstractmethod
    def load(self):
        """"""
        pass

    @abstractmethod
    def test_train_split(self, ratio: float):
        pass


class ImplicitFeedbackDataLoader(AbstractDataLoader):
    """
    We want to load the data as an optimized form of sparse matrices
    """

    LIMIT_TO_LOAD_IN_MEMORY = 1_000_000_000_000

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

        self.id_to_item_bmap = SerialBidirectionalMapper()  # bijective mapping
        self.id_to_user_bmap = SerialBidirectionalMapper()
        self.data_by_user_id = SerialUnidirectionalMapper()  # subjective mapping
        self.data_by_item_id = SerialUnidirectionalMapper()

    def load(
        self,
        *,
        file_path: str,
        user_header: str,
        item_header: str,
        data_headers: Iterable = None,
        data_constructor: Callable = None,
        limit: int = None,
        verbose: bool = False,
    ) -> None:

        indexed_count = 0

        try:
            with open(file_path, mode="r", newline="") as csvfile:
                for line_count, line in enumerate(DictReader(csvfile)):
                    user, item = line[self._user_header], line[self._item_header]
                    # Unlikely in this dataset but better have this guard
                    if not all([user, item]):
                        logger.log(
                            f"Cannot process the line {line_count}, cause expects `user` and `item` to be defined but got {user} and {item} for them respectively, skipping..."
                        )
                        continue

                    user_id = self.id_to_user_bmap.inverse[user]
                    item_id = self.id_to_item_bmap.inverse[item]

                    if user_id is None:
                        # This user is a new one, so add it
                        self.id_to_user_bmap.add(user)

                    if item_id is None:
                        # This item is a new one, so add it
                        self.id_to_item_bmap.add(item)

                    # Add the data without the useless items for indexing
                    data = self._construct_data(line, data_headers)

                    self.data_by_user_id.add(data=data, key=user_id)
                    self.data_by_item_id.add(data=data, key=item_id)

                    if verbose:
                        logger.log(
                            f"Load the line {indexed_count} of {file_path} successfully"
                        )

                    indexed_count += 1
                    if indexed_count == self._limit:
                        logger.log(
                            f"Limit of entries (.i.e {limit}) to load has been reached. Exiting without loading the rest... "
                        )
                        break

            # Return the created instance
            return self
        except (FileNotFoundError, KeyError) as exc:
            logger.log(
                f"Cannot create `InMemory2DIndexer` instance from the given `file_path` .i.e {file_path}. Attempt failed with exception {exc}"
            )
            raise

    @abstractmethod
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
