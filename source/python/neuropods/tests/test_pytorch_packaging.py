#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import shutil
import unittest
from tempfile import mkdtemp
from testpath.tempdir import TemporaryDirectory

from neuropods.backends.pytorch.packager import create_pytorch_neuropod
from neuropods.utils.env_utils import create_virtualenv

ADDITION_MODEL_SOURCE = """
import torch
import torch.nn as nn

class AdditionModel(nn.Module):
    def forward(self, x, y):
        return {
            "out": x + y
        }

def get_model(_):
    return AdditionModel()
"""


class TestPytorchPackaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse one virtualenv for all tests
        cls._virtualenv = mkdtemp()
        create_virtualenv(cls._virtualenv, ["torch", "numpy"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._virtualenv)

    def package_simple_addition_model(self, do_fail=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")
            model_code_dir = os.path.join(test_dir, "model_code")
            os.makedirs(model_code_dir)

            with open(os.path.join(model_code_dir, "addition_model.py"), "w") as f:
                f.write(ADDITION_MODEL_SOURCE)

            # `create_pytorch_neuropod` runs inference with the test data immediately
            # after creating the neuropod. Raises a ValueError if the model output
            # does not match the expected output.
            create_pytorch_neuropod(
                neuropod_path=neuropod_path,
                model_name="addition_model",
                data_paths=[],
                code_path_spec=[{
                    "python_root": model_code_dir,
                    "dirs_to_package": [
                        ""  # Package everything in the python_root
                    ],
                }],
                entrypoint_package="addition_model",
                entrypoint="get_model",
                input_spec=[
                    {"name": "x", "dtype": "float32", "shape": (None,)},
                    {"name": "y", "dtype": "float32", "shape": (None,)},
                ],
                output_spec=[
                    {"name": "out", "dtype": "float32", "shape": (None,)},
                ],
                test_input_data={
                    "x": np.arange(5),
                    "y": np.arange(5),
                },
                test_expected_out={
                    "out": np.zeros(5) if do_fail else np.arange(5) + np.arange(5)
                },
                test_deps=['torch', 'numpy'],
                test_virtualenv=self._virtualenv,
            )

    def test_simple_addition_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_simple_addition_model()

    def test_simple_addition_model_failure(self):
        # Tests a case where the output does not match the expected output
        with self.assertRaises(ValueError):
            self.package_simple_addition_model(do_fail=True)

if __name__ == '__main__':
    unittest.main()
