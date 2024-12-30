import unittest
from datetime import datetime
from unittest.mock import patch
from src.utils import convert_flat_dict_to_string


class TestConvertFlatDictToString(unittest.TestCase):
    @patch("src.utils.datetime")
    def test_with_timestamp(self, mock_datetime):
        # Mock datetime to ensure consistent timestamp
        mock_datetime.now.return_value = datetime(2024, 12, 30, 15, 24, 33)
        mock_datetime.strftime = datetime.strftime

        kwargs = {"lambda": 0.1, "gamma": 0.5, "tau": 0.2}
        result = convert_flat_dict_to_string(
            input_dict=kwargs, prefix="checkpoint", extension="pt", timestamp=True
        )
        expected = "checkpoint_20241230-152433_lambda0.1_gamma0.5_tau0.2.pt"
        self.assertEqual(result, expected)

    def test_without_timestamp(self):
        kwargs = {"width": 800, "height": 600}
        result = convert_flat_dict_to_string(
            input_dict=kwargs, prefix="plot", extension="svg", timestamp=False
        )
        expected = "plot_width800_height600.svg"
        self.assertEqual(result, expected)

    def test_empty_dict(self):
        result = convert_flat_dict_to_string(
            input_dict={}, prefix="empty", extension="txt", timestamp=True
        )
        # The result should only include the prefix, timestamp, and extension
        self.assertTrue(result.startswith("empty_"))
        self.assertTrue(result.endswith(".txt"))
        self.assertEqual(result.count("_"), 1)  # Only prefix and timestamp

    @patch("src.utils.datetime")
    def test_no_prefix_or_extension(self, mock_datetime):
        # Mock datetime for predictable timestamp
        mock_datetime.now.return_value = datetime(2024, 12, 30, 15, 24, 33)
        mock_datetime.strftime = datetime.strftime

        kwargs = {"key1": "value1", "key2": "value2"}
        result = convert_flat_dict_to_string(input_dict=kwargs, timestamp=True)
        expected = "20241230-152433_key1value1_key2value2"
        self.assertEqual(result, expected)

    def test_no_timestamp_no_prefix_no_extension(self):
        kwargs = {"key1": "value1", "key2": "value2"}
        result = convert_flat_dict_to_string(input_dict=kwargs, timestamp=False)
        expected = "key1value1_key2value2"
        self.assertEqual(result, expected)

    def test_special_characters_in_prefix_and_extension(self):
        kwargs = {"key": "value"}
        result = convert_flat_dict_to_string(
            input_dict=kwargs,
            prefix="my:prefix",
            extension="data-file",
            timestamp=False,
        )
        expected = "my:prefix_keyvalue.data-file"
        self.assertEqual(result, expected)

    def test_non_primitive_values_in_dict(self):
        kwargs = {"key1": [1, 2, 3], "key2": {"nested": "dict"}}
        with self.assertRaises(TypeError):
            convert_flat_dict_to_string(input_dict=kwargs)

    def test_empty_prefix_and_extension(self):
        kwargs = {"key": "value"}
        result = convert_flat_dict_to_string(input_dict=kwargs, timestamp=False)
        expected = "keyvalue"
        self.assertEqual(result, expected)
