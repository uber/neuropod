#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import torch
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropods.backends.torchscript.packager import create_torchscript_neuropod


class AdditionModel(torch.jit.ScriptModule):
    """
    A simple addition model
    """
    @torch.jit.script_method
    def forward(self, x, y):
        # dicts are not supported in TorchScript so we can't do this:
        # return {
        #     "out": x + y
        # }
        return (("out", x + y),)


class TestTorchScriptPackaging(unittest.TestCase):
    def package_simple_addition_model(self, do_fail=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            # `create_torchscript_neuropod` runs inference with the test data immediately
            # after creating the neuropod. Raises a ValueError if the model output
            # does not match the expected output.
            create_torchscript_neuropod(
                neuropod_path=neuropod_path,
                model_name="addition_model",
                module=AdditionModel(),
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
