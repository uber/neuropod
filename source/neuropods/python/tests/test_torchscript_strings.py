#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import torch
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropods.backends.torchscript.packager import create_torchscript_neuropod
from neuropods.tests.utils import get_string_concat_model_spec, check_strings_model

class StringsModel(torch.jit.ScriptModule):
    """
    A model that concatenates two input strings
    """
    @torch.jit.script_method
    def forward(self, x, y):
        # type: (List[str], List[str])

        # To force it to be a list of strings
        out = [""]
        for i in range(len(x)):
            f = x[i]
            s = y[i]
            out.append(f + " " + s)

        return (("out", out[1:]),)


def package_strings_model(out_dir, do_fail=False):
    neuropod_path = os.path.join(out_dir, "test_neuropod")

    # `create_torchscript_neuropod` runs inference with the test data immediately
    # after creating the neuropod. Raises a ValueError if the model output
    # does not match the expected output.
    create_torchscript_neuropod(
        neuropod_path=neuropod_path,
        model_name="strings_model",
        module=StringsModel(),
        # Get the input/output spec along with test data
        **get_string_concat_model_spec(do_fail=do_fail)
    )

    # Run some additional checks
    check_strings_model(neuropod_path)


class TestTorchScriptStrings(unittest.TestCase):
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
