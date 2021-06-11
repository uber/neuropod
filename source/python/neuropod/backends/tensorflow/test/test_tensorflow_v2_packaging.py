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
import tensorflow as tf
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropod.packagers import create_tensorflow_neuropod
from neuropod.tests.utils import (
    get_addition_model_spec,
    check_addition_model,
    requires_frameworks,
)
from neuropod.utils.eval_utils import RUN_NATIVE_TESTS


def create_tf_addition_model():
    """
    A simple addition model
    """

    class Adder(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=None, dtype=tf.float32),
                tf.TensorSpec(shape=None, dtype=tf.float32),
            ]
        )
        def add(self, x, y):
            return {"out": x + y}

    return Adder()


@requires_frameworks("tensorflow")
@unittest.skipIf(tf.__version__[0] != "2", "Skipping TF 2.x tests for TF 1.x")
class TestTensorflowV2Packaging(unittest.TestCase):
    def package_simple_addition_model(self, do_fail=False, **kwargs):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            # Get the input/output spec along with test data
            kwargs.update(get_addition_model_spec(do_fail=do_fail))

            # `create_tensorflow_neuropod` runs inference with the test data immediately
            # after creating the neuropod. Raises a ValueError if the model output
            # does not match the expected output.
            create_tensorflow_neuropod(
                neuropod_path=neuropod_path,
                model_name="addition_model",
                trackable_obj=create_tf_addition_model(),
                **kwargs
            )

            # Run some additional checks
            check_addition_model(neuropod_path)

    def test_simple_addition_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_simple_addition_model()

    def test_simple_addition_model_no_zip(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_simple_addition_model(package_as_zip=False)

    def test_simple_addition_model_failure(self):
        # Tests a case where the output does not match the expected output
        with self.assertRaises(ValueError):
            self.package_simple_addition_model(do_fail=True)

    #
    # Note: The following tests only run against the native bindings. This is okay because the
    # native bindings are the default inference implementation (and will soon be the only
    # inference implementation)
    #

    @unittest.skipIf(
        not RUN_NATIVE_TESTS,
        "Target versions are only supported by the native bindings",
    )
    def test_simple_addition_model_invalid_target_version(self):
        # Tests a case where the target platform is an invalid version or range
        with self.assertRaises(RuntimeError):
            self.package_simple_addition_model(platform_version_semver="a.b.c")

    @unittest.skipIf(
        not RUN_NATIVE_TESTS,
        "Target versions are only supported by the native bindings",
    )
    def test_simple_addition_model_no_matching_version(self):
        # Tests a case where the target platform version is not one that is
        # available
        with self.assertRaises(RuntimeError):
            self.package_simple_addition_model(platform_version_semver="0.0.1")

    @unittest.skipIf(
        not RUN_NATIVE_TESTS,
        "Target versions are only supported by the native bindings",
    )
    def test_simple_addition_model_matching_range(self):
        # Tests a case where we have an appropraite backend for the target range
        self.package_simple_addition_model(platform_version_semver="2.0.0 - 3.0.0")


if __name__ == "__main__":
    unittest.main()
