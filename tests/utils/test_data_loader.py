import unittest

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
    def test_data_loader(self):
        data_loader = DataLoader(
            file_path="./../fixtures/1000_ratings.csv",
            user_header="userId",
            item_header="movieId",
            rating_header="rating",
            verbose=False,
        )

        loaded_data = data_loader.load(approximate_train_ratio=0.8)
        self.assertIsInstance(loaded_data, LoadedDataWrapper)

        print(loaded_data.data_by_user_id__train._data)
        # print(loaded_data.data_by_item_id__train._data)
        print(loaded_data.data_by_user_id__test._data)
        # print(loaded_data.data_by_item_id__test._data)
