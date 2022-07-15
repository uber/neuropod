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

import os
import re
import numpy as np
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropod.packagers import create_python_neuropod
from neuropod.tests.utils import (
    get_addition_model_spec,
    check_addition_model,
    requires_frameworks,
)

ADDITION_MODEL_SOURCE = """
import sys

def addition_model(x, y):
    return {
        "out": x + y
    }

def get_model(_):
    # Test for https://github.com/uber/neuropod/issues/374
    assert hasattr(sys, 'argv'), "sys.argv should exist"

    return addition_model
"""

NONCONTIGUOUS_MODEL_SOURCE = """
def split(x):
    x1 = x[:, :2]
    x2 = x[:, 2:]
    return {'x1': x1, 'x2': x2}

def get_model(_):
    return split
"""


@requires_frameworks("python")
class TestPythonPackaging(unittest.TestCase):
    def package_simple_addition_model(self, test_dir, do_fail=False, **kwargs):
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
            **get_addition_model_spec(do_fail=do_fail),
            **kwargs
        )

        # Run some additional checks
        check_addition_model(neuropod_path)

    def test_simple_addition_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        with TemporaryDirectory() as test_dir:
            self.package_simple_addition_model(test_dir)

    def test_simple_addition_model_failure(self):
        # Tests a case where the output does not match the expected output
        with TemporaryDirectory() as test_dir:
            with self.assertRaises(ValueError):
                self.package_simple_addition_model(test_dir, do_fail=True)

    def test_manifest(self):
        # Packages a model and ensures the manifest contains what we expect
        TARGET = """{
            "files": {
                "0/code/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "0/code/addition_model.py": "14566dd9b13641f6482f492398cd1b567d132a08facde7b2f86cddd198f93415",
                "0/config.json": "a52f77051501f426b04f865580ee81ad1c242e47c45dcc4108c673a95f12b755",
                "config.json": "0309cf2117d887218e7edb4c918097a6a427fffcbf0a948115d32414b2b6e144"
            },
            "manifest_version": 1
        }"""

        def remove_whitespace_except_newline(text):
            space_regex = re.compile(r"[^\S\r\n]")
            return space_regex.sub("", text)

        with TemporaryDirectory() as test_dir:
            # Package the model
            self.package_simple_addition_model(
                test_dir, package_as_zip=False, persist_test_data=False
            )

            # Get the manifest path
            manifest_path = os.path.join(test_dir, "test_neuropod", "manifest.json")

            # Confirm the contents match our target
            with open(manifest_path, "r") as manifest_file:
                contents = remove_whitespace_except_newline(manifest_file.read())
                target = remove_whitespace_except_newline(TARGET)

                self.assertEqual(contents, target)

    def test_noncontiguous_array(self):
        x = np.arange(16).astype(np.int64).reshape(4, 4)

        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")
            model_code_dir = os.path.join(test_dir, "model_code")
            os.makedirs(model_code_dir)

            with open(os.path.join(model_code_dir, "splitter_model.py"), "w") as f:
                f.write(NONCONTIGUOUS_MODEL_SOURCE)

            create_python_neuropod(
                neuropod_path=neuropod_path,
                model_name="splitter",
                data_paths=[],
                code_path_spec=[
                    {
                        "python_root": model_code_dir,
                        "dirs_to_package": [
                            ""  # Package everything in the python_root
                        ],
                    }
                ],
                entrypoint_package="splitter_model",
                entrypoint="get_model",
                input_spec=[{"name": "x", "dtype": "int64", "shape": (4, 4)}],
                output_spec=[
                    {"name": "x1", "dtype": "int64", "shape": (4, 2)},
                    {"name": "x2", "dtype": "int64", "shape": (4, 2)},
                ],
                test_input_data={"x": x},
                test_expected_out={"x1": x[:, :2], "x2": x[:, 2:]},
            )


if __name__ == "__main__":
    unittest.main()
