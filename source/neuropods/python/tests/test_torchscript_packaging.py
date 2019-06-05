#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import torch
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropods.packagers import create_torchscript_neuropod
from neuropods.tests.utils import get_addition_model_spec, check_addition_model

class AdditionModel(torch.jit.ScriptModule):
    """
    A simple addition model
    """
    @torch.jit.script_method
    def forward(self, x, y):
        return {
            "out": x + y
        }

class AdditionModelDictInput(torch.jit.ScriptModule):
    """
    A simple addition model.
    If there are a large number of inputs, it may be more convenient to take the
    input as a dict rather than as individual parameters.
    """
    @torch.jit.script_method
    def forward(self, data):
        # type: (Dict[str, Tensor])
        return {
            "out": data["x"] + data["y"]
        }

class TestTorchScriptPackaging(unittest.TestCase):
    def package_simple_addition_model(self, do_fail=False):
        for model in [AdditionModel, AdditionModelDictInput]:
            with TemporaryDirectory() as test_dir:
                neuropod_path = os.path.join(test_dir, "test_neuropod")

                # `create_torchscript_neuropod` runs inference with the test data immediately
                # after creating the neuropod. Raises a ValueError if the model output
                # does not match the expected output.
                create_torchscript_neuropod(
                    neuropod_path=neuropod_path,
                    model_name="addition_model",
                    module=model(),
                    # Get the input/output spec along with test data
                    **get_addition_model_spec(do_fail=do_fail)
                )

                # Run some additional checks
                check_addition_model(neuropod_path)

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
