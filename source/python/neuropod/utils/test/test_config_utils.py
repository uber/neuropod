# Copyright (c) 2020 UATC, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from neuropod.utils.config_utils import (
    canonicalize_tensor_spec,
    validate_neuropod_config,
)


def get_valid_config():
    return {
        "name": "addition_model",
        "platform": "tensorflow",
        "input_spec": [
            {"name": "x", "dtype": "float32", "shape": (None, 2, "some_symbol")},
        ],
        "output_spec": [
            {"name": "y", "dtype": "float32", "shape": (None, 2, "some_symbol")},
        ],
        "input_tensor_device": {"x": "GPU"},
    }


class TestSpecValidation(unittest.TestCase):
    def assertSpecsEqual(self, first, second):
        self.assertEqual(len(first), len(second))

        for a, b in zip(first, second):
            self.assertDictEqual(a, b)

    def test_canonicalize_tensor_spec(self):
        INPUT = [
            {"name": "x", "dtype": "string", "shape": (None, 2, "some_symbol")},
            {"name": "y", "dtype": "double", "shape": (None, 2)},
        ]

        TARGET = [
            {"name": "x", "dtype": "string", "shape": (None, 2, "some_symbol")},
            {"name": "y", "dtype": "float64", "shape": (None, 2)},
        ]

        self.assertSpecsEqual(TARGET, canonicalize_tensor_spec(TARGET))
        self.assertSpecsEqual(TARGET, canonicalize_tensor_spec(INPUT))

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

    def test_validate_neuropod_config_invalid_device(self):
        config = get_valid_config()
        config["input_tensor_device"]["x"] = "TPU"

        with self.assertRaises(ValueError):
            validate_neuropod_config(config)

    def test_validate_neuropod_config_device_without_input(self):
        config = get_valid_config()
        config["input_tensor_device"]["y"] = "CPU"

        with self.assertRaises(ValueError):
            validate_neuropod_config(config)

    def test_validate_neuropod_config_input_without_device(self):
        config = get_valid_config()
        config["input_tensor_device"] = {}

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


if __name__ == "__main__":
    unittest.main()
