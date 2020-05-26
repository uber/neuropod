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

import numpy as np
import unittest

from neuropod.backends.neuropod_executor import validate_tensors_against_specs

TEST_SPEC = [
    {"name": "x", "dtype": "float32", "shape": (None, 2)},
    {"name": "y", "dtype": "float32", "shape": (None, 2)},
]


class TestSpecValidation(unittest.TestCase):
    def test_correct_inputs(self):
        test_input = {
            "x": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "y": np.array([[1, 2], [3, 4]], dtype=np.float32),
        }

        # Shouldn't raise a ValueError
        validate_tensors_against_specs(test_input, TEST_SPEC)

    @unittest.skip(
        "We temporary made all tensors optional, until a proper mechanism is implemented"
    )
    def test_missing_tensor(self):
        test_input = {
            "x": np.array([[1, 2], [3, 4]], dtype=np.float32),
        }

        with self.assertRaises(ValueError):
            # Missing a tensor
            validate_tensors_against_specs(test_input, TEST_SPEC)

    @unittest.skip(
        "We temporary made all tensors optional, until a proper mechanism is implemented"
    )
    def test_missing_tensors(self):
        test_input = {}

        with self.assertRaises(ValueError):
            # Missing tensors
            validate_tensors_against_specs(test_input, TEST_SPEC)

    def test_bogus_tensor_name(self):
        test_input = {
            "x": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "bogus": np.array([[1, 2], [3, 4]], dtype=np.float32),
        }

        with self.assertRaises(ValueError):
            # Missing tensors
            validate_tensors_against_specs(test_input, TEST_SPEC)

    def test_incorrect_dtype(self):
        test_input = {
            "x": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "y": np.array([[1, 2], [3, 4]], dtype=np.int32),
        }

        with self.assertRaises(ValueError):
            # Incorrect dtype
            validate_tensors_against_specs(test_input, TEST_SPEC)

    def test_invalid_num_dims(self):
        test_input = {
            "x": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "y": np.array([1, 2], dtype=np.float32),
        }

        with self.assertRaises(ValueError):
            # "y" only has one dim
            validate_tensors_against_specs(test_input, TEST_SPEC)

    def test_invalid_shape(self):
        test_input = {
            "x": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "y": np.array([[1], [3]], dtype=np.float32),
        }

        with self.assertRaises(ValueError):
            # Dim 1 of "y" is incorrect
            validate_tensors_against_specs(test_input, TEST_SPEC)

    def test_correct_symbol(self):
        test_input = {
            "x": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
            "y": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        }

        SPEC = [
            {"name": "x", "dtype": "float32", "shape": ("some_symbol", 2)},
            {"name": "y", "dtype": "float32", "shape": (None, "some_symbol")},
        ]

        # Shouldn't raise a ValueError
        validate_tensors_against_specs(test_input, SPEC)

    def test_incorrect_symbol(self):
        test_input = {
            "x": np.array([[1, 2]], dtype=np.float32),
            "y": np.array([[1, 2], [3, 4]], dtype=np.float32),
        }

        SPEC = [
            {"name": "x", "dtype": "float32", "shape": ("some_symbol", 2)},
            {"name": "y", "dtype": "float32", "shape": (None, "some_symbol")},
        ]

        with self.assertRaises(ValueError):
            # Dim 1 of y should be the same as dim 0 of x (2 != 1)
            validate_tensors_against_specs(test_input, SPEC)

    def test_invalid_shape_entry(self):
        test_input = {
            "x": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "y": np.array([[1, 2], [3, 4]], dtype=np.float32),
        }

        SPEC = [
            {"name": "x", "dtype": "float32", "shape": (True, 2)},
            {"name": "y", "dtype": "float32", "shape": (None, 2)},
        ]

        with self.assertRaises(ValueError):
            # `True` is invalid in a shape specification
            validate_tensors_against_specs(test_input, SPEC)

    def test_string_tensors(self):
        test_input = {
            "x": np.array([["some", "string", "tensor"]], dtype=np.str_),
        }

        SPEC = [
            {"name": "x", "dtype": "string", "shape": (1, 3)},
        ]

        # Shouldn't raise a ValueError
        validate_tensors_against_specs(test_input, SPEC)


if __name__ == "__main__":
    unittest.main()
