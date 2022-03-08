# Copyright (c) 2020 The Neuropod Authors
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
import sys
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropod.loader import load_neuropod
from neuropod.packagers import create_python_neuropod
from neuropod.tests.utils import requires_frameworks

# From the example at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
SKLEARN_MODEL_SOURCE = """
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3

reg = LinearRegression().fit(X, y)

def model(x):
    return {
        "out": reg.predict(x)
    }

def get_model(_):
    return model
"""


@requires_frameworks("python")
class TestPythonDeps(unittest.TestCase):
    def package_sklearn_model(self, test_dir):
        neuropod_path = os.path.join(test_dir, "test_neuropod")
        model_code_dir = os.path.join(test_dir, "model_code")
        os.makedirs(model_code_dir)

        with open(os.path.join(model_code_dir, "sklearn_model.py"), "w") as f:
            f.write(SKLEARN_MODEL_SOURCE)

        create_python_neuropod(
            neuropod_path=neuropod_path,
            model_name="sklearn_model",
            data_paths=[],
            code_path_spec=[
                {
                    "python_root": model_code_dir,
                    "dirs_to_package": [""],  # Package everything in the python_root
                }
            ],
            entrypoint_package="sklearn_model",
            entrypoint="get_model",
            input_spec=[{"name": "x", "dtype": "float64", "shape": ("batch_size", 2)}],
            output_spec=[{"name": "out", "dtype": "float64", "shape": ("batch_size",)}],
            test_input_data={"x": np.array([[3, 5]], dtype=np.float64)},
            test_expected_out={"out": np.array([16], dtype=np.float64)},
            # We define requirements this way because this test runs on python 2.7 - 3.8
            # but there isn't a single version of scikit-learn that works on all of them
            requirements="""
            # Requirements for this model
            scikit-learn=={}
            """.format(
                "0.20.0" if sys.version_info.major == 2 else "0.22.0"
            ),
        )

        return neuropod_path

    def test_python_deps(self):
        # Test that we can correctly load two different models with the same dependencies
        with TemporaryDirectory() as test_dir1:
            with TemporaryDirectory() as test_dir2:
                path1 = self.package_sklearn_model(test_dir1)
                path2 = self.package_sklearn_model(test_dir2)

                with load_neuropod(path1) as n1:
                    with load_neuropod(path2) as n2:
                        input_data = {"x": np.array([[4, 5]], dtype=np.float64)}
                        self.assertAlmostEqual(n1.infer(input_data)["out"][0], 17)
                        self.assertAlmostEqual(n2.infer(input_data)["out"][0], 17)


if __name__ == "__main__":
    unittest.main()
