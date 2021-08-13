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
import tensorflow as tf
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropod.packagers import create_tensorflow_neuropod
from neuropod.loader import load_neuropod
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
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope("some_namespace"):
            x = tf.placeholder(tf.float32, name="in_x")
            y = tf.placeholder(tf.float32, name="in_y")

            # Assigned to a variable for clarity
            out = tf.add(x, y, name="out")  # noqa: F841

    return g.as_graph_def()


def create_tf_accumulator_model():
    """
    Accumulate input x into a variable. Return the accumulated value.
    """
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope("some_namespace"):
            acc = tf.get_variable(
                "accumulator",
                initializer=tf.zeros_initializer(),
                shape=(),
                dtype=tf.float32,
            )
            x = tf.placeholder(tf.float32, name="in_x")

            assign_op = tf.assign_add(acc, x)
            with tf.control_dependencies([assign_op]):
                tf.identity(acc, name="out")
        init_op = tf.global_variables_initializer()

    return g.as_graph_def(), init_op.name


@requires_frameworks("tensorflow")
@unittest.skipIf(tf.__version__[0] == "2", "Skipping TF 1.x tests for TF 2.x")
class TestTensorflowPackaging(unittest.TestCase):
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
                graph_def=create_tf_addition_model(),
                node_name_mapping={
                    "x": "some_namespace/in_x:0",
                    "y": "some_namespace/in_y:0",
                    # The `:0` is optional
                    "out": "some_namespace/out",
                },
                **kwargs
            )

            # Run some additional checks
            check_addition_model(neuropod_path)

    def package_accumulator_model(self, neuropod_path, init_op_name_as_list):
        graph_def, init_op_name = create_tf_accumulator_model()

        # `create_tensorflow_neuropod` runs inference with the test data immediately
        # after creating the neuropod. Raises a ValueError if the model output
        # does not match the expected output.
        create_tensorflow_neuropod(
            neuropod_path=neuropod_path,
            model_name="accumulator_model",
            graph_def=graph_def,
            node_name_mapping={
                "x": "some_namespace/in_x:0",
                "out": "some_namespace/out:0",
            },
            input_spec=[{"name": "x", "dtype": "float32", "shape": ()},],
            output_spec=[{"name": "out", "dtype": "float32", "shape": ()},],
            init_op_names=[init_op_name] if init_op_name_as_list else init_op_name,
            test_input_data={"x": np.float32(5.0),},
            test_expected_out={"out": np.float32(5.0),},
        )

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

    def test_stateful_model(self):
        # `init_op` can be passed a list of strings or a string
        for init_op_name_as_list in [False, True]:
            with TemporaryDirectory() as test_dir:
                neuropod_path = os.path.join(test_dir, "test_neuropod")
                self.package_accumulator_model(neuropod_path, init_op_name_as_list)
                neuropod_obj = load_neuropod(neuropod_path)
                np.testing.assert_equal(neuropod_obj.name, "accumulator_model")
                np.testing.assert_equal(neuropod_obj.platform, "tensorflow")
                np.testing.assert_equal(
                    neuropod_obj.infer({"x": np.float32(2.0)}), {"out": 2.0}
                )
                np.testing.assert_equal(
                    neuropod_obj.infer({"x": np.float32(4.0)}), {"out": 6.0}
                )

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
        self.package_simple_addition_model(platform_version_semver="1.0.1 - 2.0.0")


if __name__ == "__main__":
    unittest.main()
