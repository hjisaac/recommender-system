import unittest
from contextlib import suppress

from ...src.utils.mapper import SerialBidirectionalMapper, SerialUnidirectionalMapper, AbstractSerialMapper


class TestSerialMapper(unittest.TestCase):
    def test_abstract_serial_mapper(self):
        class TestMapper(AbstractSerialMapper):
            def add(self, value, *args, **kwargs):
                self._data.append(value)

        mapper = TestMapper()
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
        with suppress(IndexError):
            self.assertIsNone(mapper[10])

        # Test __iter__
        self.assertListEqual([i for i in mapper], [0, 1, 2])

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
        self.assertEqual(mapper.inverse["data1"], 3)  # Last added index should take precedence

    def test_integration(self):
        """Test integration between unidirectional and bidirectional mappers."""
        uni_mapper = SerialUnidirectionalMapper()
        bi_mapper = SerialBidirectionalMapper()

        uni_mapper.add("data1")
        bi_mapper.add("data1")

        uni_mapper.add("data2", key=0)
        bi_mapper.add("data2")

        self.assertListEqual(uni_mapper[0], ["data1", "data2"])
        self.assertEqual(bi_mapper.inverse["data1"], 0)
        self.assertEqual(bi_mapper.inverse["data2"], 1)


if __name__ == "__main__":
    unittest.main()
