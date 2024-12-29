import itertools
import unittest
import pandas as pd
from unittest.mock import patch

from src.helpers.dataset_indexer import (
    AbstractDatasetIndexer,
    DatasetIndexer,
    IndexedDatasetWrapper,
)


class TestAbstractDataLoader(unittest.TestCase):

    def test_abstract_dataset_indexer(self):
        # Define a concrete subclass of AbstractDataLoader for testing
        class __ConcreteDataLoader(AbstractDatasetIndexer):
            def __init__(self, data):  # noqa
                self.data = data

            def index(self):
                return self.data

        # Create a concrete instance of __ConcreteDataLoader
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        indexer = __ConcreteDataLoader(data)

        # Test the `index` method
        self.assertEqual(indexer.index(), data)

    def test_abstract_dataset_indexer_instantiation(self):
        # Ensure that instantiating AbstractDataLoader directly raises a TypeError
        with self.assertRaises(TypeError):
            AbstractDatasetIndexer()


class TestDatasetIndexer(unittest.TestCase):
    """
    The indexer is a fundamental component of our system. It is important
    to make sure that it is working as expected. So we're trying  to test
    it seriously.
    """

    TEST_FILE_USER_COUNT = 10
    # This helps to cover more part of the code
    HIGH_COVERAGE_SEQUENCE = [0, 0, 0, 0, 0, 1]

    def test_dataset_indexer_using_non_deterministic_bernoulli(self):
        dataset_indexer = DatasetIndexer(
            file_path="./../fixtures/1000_ratings.csv",
            user_header="userId",
            item_header="movieId",
            rating_header="rating",
            limit=5000,
        )

        indexed_data = dataset_indexer.index(approximate_train_ratio=0.8)
        self.assertIsInstance(indexed_data, IndexedDatasetWrapper)

        self.assertEqual(
            list(indexed_data.id_to_user_bmap.inverse.keys()),
            [str(i + 1) for i in range(self.TEST_FILE_USER_COUNT)],
        )

        # The training and the test set should have the same dimension for users and items
        self.assertEqual(
            len(indexed_data.data_by_user_id__train),
            len(indexed_data.data_by_user_id__test),
        )
        self.assertEqual(
            len(indexed_data.data_by_item_id__train),
            len(indexed_data.data_by_item_id__test),
        )

        user_ids__train = {
            user_id
            for user_id in range(len(indexed_data.data_by_user_id__train))
            if indexed_data.data_by_user_id__train[user_id]
        }
        user_ids__test = {
            user_id
            for user_id in range(len(indexed_data.data_by_user_id__test))
            if indexed_data.data_by_user_id__test[user_id]
        }
        item_ids__train = {
            item_id
            for item_id in range(len(indexed_data.data_by_item_id__train))
            if indexed_data.data_by_item_id__train[item_id]
        }
        item_ids__test = {
            item_id
            for item_id in range(len(indexed_data.data_by_item_id__test))
            if indexed_data.data_by_item_id__test[item_id]
        }
        self.assertSetEqual(user_ids__train.intersection(user_ids__test), set())
        self.assertEqual(
            len(user_ids__train) + len(user_ids__test),
            len(indexed_data.id_to_user_bmap),
        )
        # We can have the same item shared by different users
        self.assertGreaterEqual(
            len(item_ids__train) + len(item_ids__test),
            len(indexed_data.id_to_item_bmap),
        )

    @patch("src.helpers.dataset_indexer.sample_from_bernoulli")
    def test_dataset_indexer_using_half_deterministic_bernoulli(
        self, mocked_sample_from_bernoulli
    ):
        """
        This test is not equivalent to the first one. This one by using
        the mocked version of Bernoulli distribution guaranty that certain
        parts of the codes is covered independently of the randomness encoded
        by Bernoulli distribution usage. TO BE FACTORIZED LATER OR JUST RELY
        ON SEEDED RANDOM AFTER A GOOD NUMBER OF RUN.
        """

        cycle_iter = itertools.cycle(self.HIGH_COVERAGE_SEQUENCE)
        mocked_sample_from_bernoulli.side_effect = [
            next(cycle_iter) for _ in range(self.TEST_FILE_USER_COUNT)
        ]
        dataset_indexer = DatasetIndexer(
            file_path="./../fixtures/1000_ratings.csv",
            user_header="userId",
            item_header="movieId",
            rating_header="rating",
            limit=5000,
        )

        indexed_data = dataset_indexer.index(approximate_train_ratio=0.1)
        self.assertIsInstance(indexed_data, IndexedDatasetWrapper)

        self.assertEqual(
            list(indexed_data.id_to_user_bmap.inverse.keys()),
            [str(i + 1) for i in range(self.TEST_FILE_USER_COUNT)],
        )

        self.assertEqual(
            mocked_sample_from_bernoulli.call_count, self.TEST_FILE_USER_COUNT
        )
        # The training and the test set should have the same dimension for users and items
        self.assertEqual(
            len(indexed_data.data_by_user_id__train),
            len(indexed_data.data_by_user_id__test),
        )
        self.assertEqual(
            len(indexed_data.data_by_item_id__train),
            len(indexed_data.data_by_item_id__test),
        )

        user_ids__train = {
            user_id
            for user_id in range(len(indexed_data.data_by_user_id__train))
            if indexed_data.data_by_user_id__train[user_id]
        }
        user_ids__test = {
            user_id
            for user_id in range(len(indexed_data.data_by_user_id__test))
            if indexed_data.data_by_user_id__test[user_id]
        }
        item_ids__train = {
            item_id
            for item_id in range(len(indexed_data.data_by_item_id__train))
            if indexed_data.data_by_item_id__train[item_id]
        }
        item_ids__test = {
            item_id
            for item_id in range(len(indexed_data.data_by_item_id__test))
            if indexed_data.data_by_item_id__test[item_id]
        }
        self.assertSetEqual(user_ids__train.intersection(user_ids__test), set())
        self.assertEqual(
            len(user_ids__train) + len(user_ids__test),
            len(indexed_data.id_to_user_bmap),
        )
        # We can have the same item shared by different users
        self.assertGreaterEqual(
            len(item_ids__train) + len(item_ids__test),
            len(indexed_data.id_to_item_bmap),
        )

    @patch("src.helpers.dataset_indexer.sample_from_bernoulli")
    def test_dataset_indexer_lower_approximate_ratio_parameter(
        self, mocked_sample_from_bernoulli
    ):
        cycle_iter = itertools.cycle([0, 0, 1, 0, 0, 0, 1, 0])
        mocked_sample_from_bernoulli.side_effect = [
            next(cycle_iter) for _ in range(self.TEST_FILE_USER_COUNT)
        ]

        dataset_indexer = DatasetIndexer(
            file_path="./../fixtures/1000_ratings.csv",
            user_header="userId",
            item_header="movieId",
            rating_header="rating",
        )
        indexed_data = dataset_indexer.index(approximate_train_ratio=0.8)

        user_ids__train = {
            user_id
            for user_id in range(len(indexed_data.data_by_user_id__train))
            if indexed_data.data_by_user_id__train[user_id]
        }
        user_ids__test = {
            user_id
            for user_id in range(len(indexed_data.data_by_user_id__test))
            if indexed_data.data_by_user_id__test[user_id]
        }

        self.assertLessEqual(len(user_ids__train), len(user_ids__test))

    @patch("src.helpers.dataset_indexer.sample_from_bernoulli")
    def test_dataset_indexer_higher_approximate_ratio_parameter(
        self, mocked_sample_from_bernoulli
    ):
        cycle_iter = itertools.cycle([1, 1, 1, 0, 1, 1, 1])
        mocked_sample_from_bernoulli.side_effect = [
            next(cycle_iter) for _ in range(self.TEST_FILE_USER_COUNT)
        ]

        dataset_indexer = DatasetIndexer(
            file_path="./../fixtures/1000_ratings.csv",
            user_header="userId",
            item_header="movieId",
            rating_header="rating",
        )
        indexed_data = dataset_indexer.index(approximate_train_ratio=0.8)

        user_ids__train = {
            user_id
            for user_id in range(len(indexed_data.data_by_user_id__train))
            if indexed_data.data_by_user_id__train[user_id]
        }
        user_ids__test = {
            user_id
            for user_id in range(len(indexed_data.data_by_user_id__test))
            if indexed_data.data_by_user_id__test[user_id]
        }

        self.assertGreaterEqual(len(user_ids__train), len(user_ids__test))

    def test_dataset_indexer_bmaps_contain_every_item_and_every_user(self):
        """
        Test that the DatasetIndexer's bmaps (bidirectional maps) contain
        every item and every user from the provided dataset.
        """
        # Load the CSV file to inspect its contents
        file_path = "./../../ml-32m/ratings.csv"  # "./../fixtures/1000_ratings.csv"
        limit_to_read = 2000000

        data = pd.read_csv(file_path, nrows=limit_to_read)

        items = list(set(data["movieId"].astype(str).to_list()))
        users = list(set(data["userId"].astype(str).to_list()))

        self.assertNotEqual(len(items), 0)
        self.assertNotEqual(len(users), 0)

        dataset_indexer = DatasetIndexer(
            file_path=file_path,
            user_header="userId",
            item_header="movieId",
            rating_header="rating",
            limit=limit_to_read,
        )

        indexed_data = dataset_indexer.index(approximate_train_ratio=0.8)
        self.assertIsInstance(indexed_data, IndexedDatasetWrapper)

        # Ensure that all items in the dataset are included in the item bmap
        self.assertSetEqual(
            set(items).symmetric_difference(
                indexed_data.id_to_item_bmap.inverse.keys()
            ),
            set(),
        )

        # Ensure that all users in the dataset are included in the user bmap
        self.assertSetEqual(
            set(users).symmetric_difference(
                indexed_data.id_to_user_bmap.inverse.keys()
            ),
            set(),
        )

        # Ensure that the bmap does not have any None values for the items
        self.assertNotIn(
            None, [indexed_data.id_to_item_bmap.inverse[item] for item in items]
        )
        # Ensure that the bmap does not have any None values for the users
        self.assertNotIn(
            None, [indexed_data.id_to_user_bmap.inverse[user] for user in users]
        )
