# Copyright (c) 2021 UATC, LLC
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
import six
import numpy as np
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropod.loader import load_neuropod
from neuropod.packagers import create_python_neuropod
from neuropod.tests.utils import get_addition_model_spec

ADDITION_MODEL_SOURCE = """
import sys

def addition_model(x, y):
    return {
        "out": x + y
    }

def get_model(_):
    return addition_model
"""


@unittest.skipIf(six.PY2, "Skipping asyncio test for Python 2")
class TestAsync(unittest.TestCase):
    def package_simple_addition_model(self, test_dir, do_fail=False):
        neuropod_path = os.path.join(test_dir, "test_neuropod")
        model_code_dir = os.path.join(test_dir, "model_code")
        os.makedirs(model_code_dir)

        with open(os.path.join(model_code_dir, "addition_model.py"), "w") as f:
            f.write(ADDITION_MODEL_SOURCE)

        # `create_python_neuropod` runs inference with the test data immediately
        # after creating the neuropod. Raises a ValueError if the model output
        # does not match the expected output.
        create_python_neuropod(
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
            # Get the input/output spec along with test data
            **get_addition_model_spec(do_fail=do_fail)
        )

        return neuropod_path

    def test_async_inference(self):
        # Get an event loop
        import asyncio

        loop = asyncio.get_event_loop()

        with TemporaryDirectory() as test_dir:
            # Package a model
            path = self.package_simple_addition_model(test_dir)

            # Sample input data
            input_data = {
                "x": np.array([0.5], dtype=np.float32),
                "y": np.array([1.5], dtype=np.float32),
            }

            with load_neuropod(path) as model:
                # Async infer
                result = loop.run_until_complete(model.infer_async(input_data))

                # Ensure the output is what we expect
                self.assertEqual(result["out"][0], 2.0)


if __name__ == "__main__":
    unittest.main()
