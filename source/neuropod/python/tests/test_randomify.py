#
# Uber, Inc. (c) 2018
#
import numpy as np
import os
import shutil
import unittest
from tempfile import mkdtemp

from neuropod.loader import load_neuropod
from neuropod.utils.randomify import randomify_neuropod

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
        with self.assertRaises(ValueError):
            neuropod = load_neuropod(TestSpecValidation.neuropod_path)
            neuropod.infer(
                {"bogus": np.asarray([[1.1, 2.2], [0, 1], [2, 3]], dtype=np.float32)}
            )

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            neuropod = load_neuropod(TestSpecValidation.neuropod_path)
            neuropod.infer({"in_float32_matrix": np.asarray([3], dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
