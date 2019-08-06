#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import shutil
import unittest
from tempfile import mkdtemp
from testpath.tempdir import TemporaryDirectory

from neuropods.loader import load_neuropod
from neuropods.packagers import create_python_neuropod
from neuropods.tests.utils import get_addition_model_spec, check_addition_model

DUMMY_MODEL_SOURCE = """
import numpy as np

def model(x):
    return {{
        "out": np.array([{}], dtype=np.float32)
    }}

def get_model(_):
    return model
"""


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
            code_path_spec=[{
                "python_root": model_code_dir,
                "dirs_to_package": [
                    ""  # Package everything in the python_root
                ],
            }],
            entrypoint_package="dummy_model",
            entrypoint="get_model",
            test_deps=['numpy'],
            # This runs the test in the current process instead of in a new virtualenv
            # We are using this to ensure the test will work even if the CI environment
            # is restrictive
            skip_virtualenv=True,
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

                with load_neuropod(path1) as n1:
                    with load_neuropod(path2) as n2:
                        input_data = {"x": np.array([0], dtype=np.float32)}
                        self.assertEqual(n1.infer(input_data)["out"][0], 1.0)
                        self.assertEqual(n2.infer(input_data)["out"][0], 2.0)


if __name__ == '__main__':
    unittest.main()
