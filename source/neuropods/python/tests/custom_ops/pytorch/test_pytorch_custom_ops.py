#
# Uber, Inc. (c) 2019
#

import glob
import numpy as np
import os
import shutil
import subprocess
import sys
import unittest
from tempfile import mkdtemp
from testpath.tempdir import TemporaryDirectory

from neuropods.packagers import create_pytorch_neuropod
from neuropods.tests.utils import get_addition_model_spec

ADDITION_MODEL_SOURCE = """
import torch
import torch.nn as nn
import addition_op

class AdditionModel(nn.Module):
    def forward(self, x, y):
        return {
            "out": addition_op.forward(torch.from_numpy(x), torch.from_numpy(y)).numpy()
        }

def get_model(_):
    return AdditionModel()
"""


class TestPytorchPackaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the custom op
        current_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.check_call([sys.executable, "setup.py", "build"], cwd=current_dir)
        cls.custom_op_path = glob.glob(os.path.join(current_dir, "build", "lib*", "addition_op.so"))[0]

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
                test_deps=['torch', 'numpy'],
                custom_ops=[self.custom_op_path],
                # This runs the test in the current process instead of in a new virtualenv
                # We are using this to ensure the test will work even if the CI environment
                # is restrictive
                skip_virtualenv=True,
                # Get the input/output spec along with test data
                **get_addition_model_spec(do_fail=do_fail)
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
