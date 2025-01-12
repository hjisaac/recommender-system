import numpy as np
import dill as pickle
from functools import partial
from datetime import datetime

sample_from_bernoulli = partial(np.random.binomial, n=1)


class cached_class_property:  # noqa
    """
    A decorator to define a class-level property that caches its value
    upon first access, avoiding re-computation on later calls.
    """

    def __init__(self, func):
        self.func = func
        self._cache_name = f"_{func.__name__}_cache"

    def __get__(self, instance, cls):
        if not hasattr(cls, self._cache_name):
            setattr(cls, self._cache_name, self.func(cls))
        return getattr(cls, self._cache_name)


def convert_flat_dict_to_string(
    input_dict: dict, prefix="", extension="", timestamp=True
):
    """
    Convert a flat dictionary into a string with optional prefix, extension,
    and timestamp. Useful for creating filenames or identifiers based on
    configuration.

    Parameters:
        input_dict (dict): A flat dictionary containing key-value pairs with primitive types.
        prefix (str): Optional prefix to add at the beginning of the string.
        extension (str): Optional file extension to add at the end of the string.
        timestamp (bool): Whether to include a timestamp in the string.

    Returns:
        str: Generated string with optional timestamp, prefix, and dictionary content.
    """
    prefix = prefix or ""

    extension = extension.removeprefix(".")
    dotted_extension = f".{extension}" if extension else ""

    # Start constructing the string
    string_parts = []
    if timestamp:
        string_parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Add the key-value pairs from input_dict
    for key, value in input_dict.items():
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(
                f"Value for key '{key}' must be a primitive type (str, int, float, bool)."
            )
        string_parts.append(f"{key}{value}")

    if prefix:
        string_parts.insert(0, prefix)

    joined_string = "_".join(filter(None, string_parts))

    return f"{joined_string}{dotted_extension}" if dotted_extension else joined_string


def vocabulary_based_one_hot_encode(
    words: list, vocabulary: list, np_array: bool = True
) -> list | np.ndarray:
    """
    One-hot encodes a list of word based on the provided vocabulary.

    Args:
        words (list): Words to encode.
        vocabulary (list): List of unique words for encoding.
        np_array (bool): Return as NumPy array if True, else return as a list.

    Returns:
        list or np.ndarray: One-hot encoded vector.
    """
    # Initialize the encoded vector based on the `np_array` flag
    encoded = np.array([0] * len(vocabulary)) if np_array else [0] * len(vocabulary)

    for w in words:
        try:
            index = vocabulary.index(w)
            encoded[index] = 1
        except ValueError:
            continue

    return encoded


# Added so that it can be used for debugging
def load_pickle(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            raise


def save_pickle(state, file_path):
    try:
        with open(file_path, "wb") as f:
            pickle.dump(state, f)
    except Exception:
        raise