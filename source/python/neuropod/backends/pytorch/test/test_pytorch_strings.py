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

import os
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropod.packagers import create_pytorch_neuropod
from neuropod.tests.utils import (
    get_string_concat_model_spec,
    check_strings_model,
    requires_frameworks,
)

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


@requires_frameworks("python")
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
        code_path_spec=[
            {
                "python_root": model_code_dir,
                "dirs_to_package": [""],  # Package everything in the python_root
            }
        ],
        entrypoint_package="strings_model",
        entrypoint="get_model",
        requirements="""
        torch==1.4.0
        """,
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


if __name__ == "__main__":
    unittest.main()
