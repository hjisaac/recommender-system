import unittest

from src.utils.mapper import (
    SerialBidirectionalMapper,
    SerialUnidirectionalMapper,
    AbstractSerialMapper,
)


class TestAbstractSerialMapper(unittest.TestCase):

    def test_abstract_serial_mapper(self):
        class __ConcreteMapper(AbstractSerialMapper):
            def add(self, value, *args, **kwargs):
                self._data.append(value)

        mapper = __ConcreteMapper()
        mapper.add(1)
        mapper.add(2)
        mapper.add(3)

        # Test __len__
        self.assertEqual(len(mapper), 3)

        # Test __getitem__ with valid indices
        self.assertEqual(mapper[0], 1)
        self.assertEqual(mapper[1], 2)
        self.assertEqual(mapper[2], 3)

        # Test __getitem__ with invalid index
        self.assertIsNone(mapper[10])

        # Test __iter__
        self.assertListEqual([i for i in mapper], [0, 1, 2])


class TestUnidirectionalMapper(unittest.TestCase):
    def test_serial_unidirectional_mapper(self):
        mapper = SerialUnidirectionalMapper()

        # Test adding data with no key
        mapper.add("data1")
        mapper.add("data2")
        self.assertEqual(len(mapper), 2)
        self.assertListEqual(mapper[0], ["data1"])
        self.assertListEqual(mapper[1], ["data2"])

        # Test adding data to a specific key
        mapper.add("data3", key=0)
        self.assertListEqual(mapper[0], ["data1", "data3"])

        # Test adding NOTHING
        mapper.add(SerialUnidirectionalMapper.NOTHING)
        self.assertEqual(len(mapper), 3)
        self.assertListEqual(mapper[2], [])


class TestSerialBidirectionalMapper(unittest.TestCase):
    def test_serial_bidirectional_mapper(self):
        mapper = SerialBidirectionalMapper()

        # Test adding data
        mapper.add("data1")
        mapper.add("data2")
        mapper.add("data3")

        self.assertEqual(len(mapper), 3)
        self.assertEqual(mapper[0], "data1")
        self.assertEqual(mapper[1], "data2")
        self.assertEqual(mapper[2], "data3")

        # Test inverse mapping
        self.assertEqual(mapper.inverse["data1"], 0)
        self.assertEqual(mapper.inverse["data2"], 1)
        self.assertEqual(mapper.inverse["data3"], 2)
        self.assertIsNone(mapper.inverse["data4"])  # Non-existent key

        # Test overwriting inverse data
        mapper.add("data1")  # Add duplicate
        self.assertEqual(
            mapper.inverse["data1"], 3
        )  # Last added index should take precedence


if __name__ == "__main__":
    unittest.main()
