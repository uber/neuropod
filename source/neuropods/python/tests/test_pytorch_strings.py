#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropods.packagers import create_pytorch_neuropod
from neuropods.tests.utils import get_string_concat_model_spec, check_strings_model

STRINGS_MODEL_SOURCE = """
import numpy as np
import torch
import torch.nn as nn

class StringsModel(nn.Module):
    def forward(self, x, y):
        return {
            "out": np.array([f + " " + s for f, s in zip(x, y)])
        }

def get_model(_):
    return StringsModel()
"""


def package_strings_model(out_dir, do_fail=False):
    neuropod_path = os.path.join(out_dir, "test_neuropod")
    model_code_dir = os.path.join(out_dir, "model_code")
    os.makedirs(model_code_dir)

    with open(os.path.join(model_code_dir, "strings_model.py"), "w") as f:
        f.write(STRINGS_MODEL_SOURCE)

    # `create_pytorch_neuropod` runs inference with the test data immediately
    # after creating the neuropod. Raises a ValueError if the model output
    # does not match the expected output.
    create_pytorch_neuropod(
        neuropod_path=neuropod_path,
        model_name="strings_model",
        data_paths=[],
        code_path_spec=[{
            "python_root": model_code_dir,
            "dirs_to_package": [
                ""  # Package everything in the python_root
            ],
        }],
        entrypoint_package="strings_model",
        entrypoint="get_model",
        test_deps=['torch', 'numpy'],
        # This runs the test in the current process instead of in a new virtualenv
        # We are using this to ensure the test will work even if the CI environment
        # is restrictive
        skip_virtualenv=True,
        # Get the input/output spec along with test data
        **get_string_concat_model_spec(do_fail=do_fail)
    )

    # Run some additional checks
    check_strings_model(neuropod_path)


class TestPytorchPackaging(unittest.TestCase):
    def test_strings_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        with TemporaryDirectory() as test_dir:
            package_strings_model(test_dir)

    def test_strings_model_failure(self):
        # Tests a case where the output does not match the expected output
        with TemporaryDirectory() as test_dir:
            with self.assertRaises(ValueError):
                package_strings_model(test_dir, do_fail=True)


if __name__ == '__main__':
    unittest.main()
