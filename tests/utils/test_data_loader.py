import itertools
import unittest
from unittest.mock import patch

from src.utils.data_loader import AbstractDataLoader, DataLoader, LoadedDataWrapper


class TestAbstractDataLoader(unittest.TestCase):

    def test_data_loader(self):
        # Define a concrete subclass of AbstractDataLoader for testing
        class __ConcreteDataLoader(AbstractDataLoader):
            def __init__(self, data):  # noqa
                self.data = data

            def load(self):
                return self.data

            def test_train_split(self, ratio: float):
                split_index = int(len(self.data) * ratio)
                return self.data[:split_index], self.data[split_index:]

        # Create a concrete instance of __ConcreteDataLoader
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        loader = __ConcreteDataLoader(data)

        # Test the `load` method
        self.assertEqual(loader.load(), data)

        # Test the `test_train_split` method
        train, test = loader.test_train_split(0.8)
        self.assertEqual(train, [1, 2, 3, 4, 5, 6, 7, 8])  # 80% training data
        self.assertEqual(test, [9, 10])  # 20% test data

    def test_abstract_methods(self):
        # Ensure that instantiating AbstractDataLoader directly raises a TypeError
        with self.assertRaises(TypeError):
            AbstractDataLoader()


class TestDataLoader(unittest.TestCase):
    TEST_FILE_USER_COUNT = 10
    # This helps to cover more part of the code
    HIGH_COVERAGE_SEQUENCE = [0, 0, 0, 0, 0, 1]

    def test_data_loader_using_non_deterministic_bernoulli(self):
        data_loader = DataLoader(
            file_path="./../fixtures/1000_ratings.csv",
            user_header="userId",
            item_header="movieId",
            rating_header="rating",
            limit=5000,
            verbose=False,
        )

        loaded_data = data_loader.load(approximate_train_ratio=000.1)
        self.assertIsInstance(loaded_data, LoadedDataWrapper)

        self.assertEqual(
            list(loaded_data.id_to_user_bmap.inverse.keys()),
            [str(i + 1) for i in range(self.TEST_FILE_USER_COUNT)],
        )

        # The training and the test set should have the same dimension for users and items
        self.assertEqual(
            len(loaded_data.data_by_user_id__train),
            len(loaded_data.data_by_user_id__test),
        )
        self.assertEqual(
            len(loaded_data.data_by_item_id__train),
            len(loaded_data.data_by_item_id__test),
        )

        user_ids__train = {
            user_id
            for user_id in range(len(loaded_data.data_by_user_id__train))
            if loaded_data.data_by_user_id__train[user_id]
        }
        user_ids__test = {
            user_id
            for user_id in range(len(loaded_data.data_by_user_id__test))
            if loaded_data.data_by_user_id__test[user_id]
        }
        item_ids__train = {
            item_id
            for item_id in range(len(loaded_data.data_by_item_id__train))
            if loaded_data.data_by_item_id__train[item_id]
        }
        item_ids__test = {
            item_id
            for item_id in range(len(loaded_data.data_by_item_id__test))
            if loaded_data.data_by_item_id__test[item_id]
        }
        self.assertSetEqual(user_ids__train.intersection(user_ids__test), set())
        self.assertEqual(
            len(user_ids__train) + len(user_ids__test), len(loaded_data.id_to_user_bmap)
        )
        # We can have the same item shared by different users
        self.assertGreaterEqual(
            len(item_ids__train) + len(item_ids__test), len(loaded_data.id_to_item_bmap)
        )

    @patch("src.utils.data_loader.sample_from_bernoulli")
    def test_data_loader_using_half_deterministic_bernoulli(
        self, mocked_sample_from_bernoulli
    ):
        """
        This test is not equivalent to the first one. This one by using
        the mocked version of Bernoulli distribution guaranty that certain
        parts of the codes is covered independently of the randomness encoded
        by Bernoulli distribution usage. TO BE FACTORIZED LATER.
        """

        cycle_iter = itertools.cycle(self.HIGH_COVERAGE_SEQUENCE)
        mocked_sample_from_bernoulli.side_effect = [
            next(cycle_iter) for _ in range(self.TEST_FILE_USER_COUNT)
        ]
        data_loader = DataLoader(
            file_path="./../fixtures/1000_ratings.csv",
            user_header="userId",
            item_header="movieId",
            rating_header="rating",
            limit=5000,
            verbose=False,
        )

        loaded_data = data_loader.load(approximate_train_ratio=000.1)
        self.assertIsInstance(loaded_data, LoadedDataWrapper)

        self.assertEqual(
            list(loaded_data.id_to_user_bmap.inverse.keys()),
            [str(i + 1) for i in range(self.TEST_FILE_USER_COUNT)],
        )

        self.assertEqual(
            mocked_sample_from_bernoulli.call_count, self.TEST_FILE_USER_COUNT
        )
        # The training and the test set should have the same dimension for users and items
        self.assertEqual(
            len(loaded_data.data_by_user_id__train),
            len(loaded_data.data_by_user_id__test),
        )
        self.assertEqual(
            len(loaded_data.data_by_item_id__train),
            len(loaded_data.data_by_item_id__test),
        )

        user_ids__train = {
            user_id
            for user_id in range(len(loaded_data.data_by_user_id__train))
            if loaded_data.data_by_user_id__train[user_id]
        }
        user_ids__test = {
            user_id
            for user_id in range(len(loaded_data.data_by_user_id__test))
            if loaded_data.data_by_user_id__test[user_id]
        }
        item_ids__train = {
            item_id
            for item_id in range(len(loaded_data.data_by_item_id__train))
            if loaded_data.data_by_item_id__train[item_id]
        }
        item_ids__test = {
            item_id
            for item_id in range(len(loaded_data.data_by_item_id__test))
            if loaded_data.data_by_item_id__test[item_id]
        }
        self.assertSetEqual(user_ids__train.intersection(user_ids__test), set())
        self.assertEqual(
            len(user_ids__train) + len(user_ids__test), len(loaded_data.id_to_user_bmap)
        )
        # We can have the same item shared by different users
        self.assertGreaterEqual(
            len(item_ids__train) + len(item_ids__test), len(loaded_data.id_to_item_bmap)
        )
