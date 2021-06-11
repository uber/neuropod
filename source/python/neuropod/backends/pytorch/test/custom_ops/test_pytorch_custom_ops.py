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

import glob
import os
import shutil
import subprocess
import sys
import unittest

import torch
from testpath.tempdir import TemporaryDirectory

from neuropod.packagers import create_pytorch_neuropod
from neuropod.tests.utils import get_addition_model_spec, requires_frameworks
from neuropod.utils.hash_utils import sha256sum

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


def build_op(workdir):
    subprocess.check_call([sys.executable, "setup.py", "build"], cwd=workdir)
    return glob.glob(os.path.join(workdir, "build", "lib*", "addition_op.so"))[0]


@requires_frameworks("python")
@unittest.skipIf(
    not torch.__version__.startswith("1.4.0"),
    "Skipping custom op test for torch != 1.4.0",
)
class TestPytorchPackaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the custom op
        current_dir = os.path.dirname(os.path.abspath(__file__))

        cls.custom_op_path = build_op(os.path.join(current_dir, "addition_op_1"))

        # For testing loading of a custom op multiple times
        cls.second_custom_op = os.path.join(current_dir, "addition_op_copy.so")
        shutil.copyfile(cls.custom_op_path, cls.second_custom_op)

        # This op is named the same thing as the above op, but has a different
        # implementation
        cls.other_op_path = build_op(os.path.join(current_dir, "addition_op_2"))

    def package_simple_addition_model(self, test_dir, custom_ops, do_fail=False):
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
            code_path_spec=[
                {
                    "python_root": model_code_dir,
                    "dirs_to_package": [""],  # Package everything in the python_root
                }
            ],
            entrypoint_package="addition_model",
            entrypoint="get_model",
            custom_ops=custom_ops,
            requirements="""
            torch==1.4.0
            """,
            # Get the input/output spec along with test data
            **get_addition_model_spec(do_fail=do_fail)
        )

        return neuropod_path

    def test_simple_addition_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        with TemporaryDirectory() as test_dir:
            self.package_simple_addition_model(
                test_dir, custom_ops=[self.custom_op_path, self.second_custom_op]
            )

    def test_simple_addition_model_failure(self):
        # Tests a case where the output does not match the expected output
        with TemporaryDirectory() as test_dir:
            with self.assertRaises(ValueError):
                self.package_simple_addition_model(
                    test_dir, do_fail=True, custom_ops=[self.custom_op_path]
                )

    def test_consistent_hash(self):
        # Packages the same model twice and ensures it has the same hash
        shas = []
        for i in range(2):
            with TemporaryDirectory() as test_dir:
                self.package_simple_addition_model(
                    test_dir, custom_ops=[self.custom_op_path, self.second_custom_op]
                )

                neuropod_path = os.path.join(test_dir, "test_neuropod")
                shas.append(sha256sum(neuropod_path))

        self.assertEqual(shas[0], shas[1])


if __name__ == "__main__":
    unittest.main()
