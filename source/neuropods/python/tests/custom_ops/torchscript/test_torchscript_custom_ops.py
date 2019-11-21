#
# Uber, Inc. (c) 2019
#

import glob
import os
import subprocess
import sys
import torch
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropods.packagers import create_torchscript_neuropod
from neuropods.tests.utils import get_addition_model_spec, check_addition_model


class CustomOpModel(torch.jit.ScriptModule):
    """
    A simple addition model that uses a custom op
    """

    @torch.jit.script_method
    def forward(self, x, y):
        return {"out": torch.ops.neuropod_test_ops.add(x, y)}


class TestTorchScriptCustomOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the custom op
        current_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.check_call([sys.executable, "setup.py", "build"], cwd=current_dir)
        cls.custom_op_path = glob.glob(
            os.path.join(current_dir, "build", "lib*", "addition_op.so")
        )[0]

        # Load the op
        torch.ops.load_library(cls.custom_op_path)

    def package_simple_addition_model(self, do_fail=False):
        for model in [CustomOpModel]:
            with TemporaryDirectory() as test_dir:
                neuropod_path = os.path.join(test_dir, "test_neuropod")

                # `create_torchscript_neuropod` runs inference with the test data immediately
                # after creating the neuropod. Raises a ValueError if the model output
                # does not match the expected output.
                create_torchscript_neuropod(
                    neuropod_path=neuropod_path,
                    model_name="addition_model",
                    module=model(),
                    custom_ops=[self.custom_op_path],
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


if __name__ == "__main__":
    unittest.main()
