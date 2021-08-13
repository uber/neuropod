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
import os
import six
import shutil
import unittest
import tensorflow as tf

from tempfile import mkdtemp

from neuropod.loader import load_neuropod
from neuropod.utils.randomify import randomify_neuropod
from neuropod.tests.utils import requires_frameworks

input_spec = [
    {"name": "in_string_vector", "dtype": "string", "shape": ("N",)},
    {"name": "in_int_matrix", "dtype": "int32", "shape": ("N", "M")},
    {"name": "in_float32_matrix", "dtype": "float32", "shape": ("N", None)},
    {"name": "in_float64_scalar", "dtype": "float64", "shape": ()},
]

output_spec = [
    {"name": "out_string_vector", "dtype": "string", "shape": ("N",)},
    {"name": "out_int_matrix", "dtype": "int32", "shape": ("N", "M")},
    {"name": "out_float_matrix", "dtype": "int32", "shape": ("N", None)},
    {"name": "out_float_scalar", "dtype": "float32", "shape": ()},
]


@requires_frameworks("tensorflow")
@unittest.skipIf(tf.__version__[0] == "2", "Skipping TF 1.x tests for TF 2.x")
class TestSpecValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = mkdtemp()

        cls.neuropod_path = os.path.join(cls.tmpdir, "test_stub_neuropod")
        randomify_neuropod(cls.neuropod_path, input_spec, output_spec)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_no_inputs(self):
        neuropod = load_neuropod(TestSpecValidation.neuropod_path)
        result = neuropod.infer({})
        self.assertGreater(result["out_string_vector"].shape[0], 0)
        self.assertEqual(
            result["out_string_vector"].shape[0], result["out_int_matrix"].shape[0]
        )
        self.assertGreater(result["out_float_matrix"].shape[0], 0)

        # TODO(vip): Fix native scalars
        # self.assertTrue(np.isscalar(result["out_float_scalar"]))

    def test_some_inputs(self):
        neuropod = load_neuropod(TestSpecValidation.neuropod_path)
        result = neuropod.infer(
            {
                "in_float32_matrix": np.asarray(
                    [[1.1, 2.2], [0, 1], [2, 3]], dtype=np.float32
                )
            }
        )
        self.assertGreater(result["out_string_vector"].shape[0], 0)

    def test_invalid_input_name(self):
        with six.assertRaisesRegex(
            self, (ValueError, RuntimeError), "are not found in the input spec"
        ):
            neuropod = load_neuropod(TestSpecValidation.neuropod_path)
            neuropod.infer(
                {"bogus": np.asarray([[1.1, 2.2], [0, 1], [2, 3]], dtype=np.float32)}
            )

    def test_invalid_shape(self):
        with six.assertRaisesRegex(
            self,
            (ValueError, RuntimeError),
            "in the input spec is expected to have 2 dimensions, but had 1",
        ):
            neuropod = load_neuropod(TestSpecValidation.neuropod_path)
            neuropod.infer({"in_float32_matrix": np.asarray([3], dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
