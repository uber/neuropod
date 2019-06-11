#
# Uber, Inc. (c) 2018
#

import unittest

from neuropods.backends.config_utils import canonicalize_tensor_spec, validate_neuropod_config


def get_valid_config():
    return {
        "name": "addition_model",
        "platform": "tensorflow",
        "input_spec": [
            {"name": "x", "dtype": "float32", "shape": (None, 2, "some_symbol"), "device": "GPU"},
        ],
        "output_spec": [
            {"name": "y", "dtype": "float32", "shape": (None, 2, "some_symbol")},
        ]
    }


class TestSpecValidation(unittest.TestCase):
    def assertSpecsEqual(self, first, second):
        self.assertEqual(len(first), len(second))

        for a, b in zip(first, second):
            self.assertDictEqual(a, b)

    def test_canonicalize_tensor_spec(self):
        INPUT = [
            {"name": "x", "dtype": "string", "shape": (None, 2, "some_symbol")},
            {"name": "y", "dtype": "double", "shape": (None, 2), "device": "CPU"},
        ]

        TARGET = [
            {"name": "x", "dtype": "string", "shape": (None, 2, "some_symbol"), "device": "GPU"},
            {"name": "y", "dtype": "float64", "shape": (None, 2), "device": "CPU"},
        ]

        TARGET_NO_DEFAULT = [
            {"name": "x", "dtype": "string", "shape": (None, 2, "some_symbol")},
            {"name": "y", "dtype": "float64", "shape": (None, 2), "device": "CPU"},
        ]

        self.assertSpecsEqual(TARGET, canonicalize_tensor_spec(TARGET, default_device="CPU"))
        self.assertSpecsEqual(TARGET, canonicalize_tensor_spec(INPUT, default_device="GPU"))
        self.assertSpecsEqual(TARGET_NO_DEFAULT, canonicalize_tensor_spec(INPUT, default_device=None))

    def test_validate_neuropod_config(self):
        validate_neuropod_config(get_valid_config())

    def test_validate_neuropod_config_invalid_name(self):
        config = get_valid_config()
        config["name"] = True

        with self.assertRaises(ValueError):
            validate_neuropod_config(config)

    def test_validate_neuropod_config_invalid_platform(self):
        config = get_valid_config()
        config["platform"] = 5

        with self.assertRaises(ValueError):
            validate_neuropod_config(config)

    def test_validate_neuropod_config_invalid_spec_device(self):
        config = get_valid_config()
        config["input_spec"][0]["device"] = "TPU"

        with self.assertRaises(ValueError):
            validate_neuropod_config(config)

    def test_validate_neuropod_config_invalid_spec_dtype(self):
        config = get_valid_config()
        config["input_spec"][0]["dtype"] = "complex128"

        with self.assertRaises(ValueError):
            validate_neuropod_config(config)

    def test_validate_neuropod_config_invalid_spec_name(self):
        config = get_valid_config()
        config["input_spec"][0]["name"] = 5

        with self.assertRaises(ValueError):
            validate_neuropod_config(config)

    def test_validate_neuropod_config_invalid_spec_shape(self):
        config = get_valid_config()
        config["input_spec"][0]["shape"] = "123"

        with self.assertRaises(ValueError):
            validate_neuropod_config(config)

    def test_validate_neuropod_config_invalid_spec_shape_element(self):
        config = get_valid_config()
        config["input_spec"][0]["shape"] = (None, 2, "some_symbol", True)

        with self.assertRaises(ValueError):
            validate_neuropod_config(config)


if __name__ == '__main__':
    unittest.main()
