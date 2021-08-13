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

import numpy as np
import os
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropod.loader import load_neuropod
from neuropod.packagers import create_python_neuropod
from neuropod.tests.utils import requires_frameworks

DUMMY_MODEL_SOURCE = """
import numpy as np

def model(x):
    return {{
        "out": np.array([{}], dtype=np.float32)
    }}

def get_model(_):
    return model
"""


@requires_frameworks("python")
class TestPythonIsolation(unittest.TestCase):
    def package_dummy_model(self, test_dir, target):
        neuropod_path = os.path.join(test_dir, "test_neuropod")
        model_code_dir = os.path.join(test_dir, "model_code")
        os.makedirs(model_code_dir)

        with open(os.path.join(model_code_dir, "dummy_model.py"), "w") as f:
            f.write(DUMMY_MODEL_SOURCE.format(target))

        # `create_python_neuropod` runs inference with the test data immediately
        # after creating the neuropod. Raises a ValueError if the model output
        # does not match the expected output.
        create_python_neuropod(
            neuropod_path=neuropod_path,
            model_name="dummy_model",
            data_paths=[],
            code_path_spec=[
                {
                    "python_root": model_code_dir,
                    "dirs_to_package": [""],  # Package everything in the python_root
                }
            ],
            entrypoint_package="dummy_model",
            entrypoint="get_model",
            input_spec=[{"name": "x", "dtype": "float32", "shape": (1,)}],
            output_spec=[{"name": "out", "dtype": "float32", "shape": (1,)}],
            test_input_data={"x": np.array([0.0], dtype=np.float32)},
            test_expected_out={"out": np.array([target], dtype=np.float32)},
        )

        return neuropod_path

    def test_model_isolation(self):
        # Test that we can correctly load two different models with the same entrypoint in the same process
        with TemporaryDirectory() as test_dir1:
            with TemporaryDirectory() as test_dir2:
                path1 = self.package_dummy_model(test_dir1, 1.0)
                path2 = self.package_dummy_model(test_dir2, 2.0)

                with load_neuropod(path1, _always_use_native=False) as n1:
                    with load_neuropod(path2, _always_use_native=False) as n2:
                        input_data = {"x": np.array([0], dtype=np.float32)}
                        self.assertEqual(n1.infer(input_data)["out"][0], 1.0)
                        self.assertEqual(n2.infer(input_data)["out"][0], 2.0)


if __name__ == "__main__":
    unittest.main()
