#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import tensorflow as tf
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropods.backends.tensorflow.packager import create_tensorflow_neuropod


def create_tf_addition_model():
    """
    A simple addition model
    """
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope("some_namespace"):
            x = tf.placeholder(tf.float32, name="in_x")
            y = tf.placeholder(tf.float32, name="in_y")

            # UATG(flake8/F841) Assigned to a variable for clarity
            out = tf.add(x, y, name="out")

    return g.as_graph_def()


class TestTensorflowPackaging(unittest.TestCase):
    def package_simple_addition_model(self, do_fail=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

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
                    "out": "some_namespace/out:0",
                },
                input_spec=[
                    {"name": "x", "dtype": "float32", "shape": (None,)},
                    {"name": "y", "dtype": "float32", "shape": (None,)},
                    {"name": "optional", "dtype": "string", "shape": (None,)},
                ],
                output_spec=[
                    {"name": "out", "dtype": "float32", "shape": (None,)},
                ],
                test_input_data={
                    "x": np.arange(5, dtype=np.float32),
                    "y": np.arange(5, dtype=np.float32),
                },
                test_expected_out={
                    "out": np.zeros(5) if do_fail else np.arange(5) + np.arange(5)
                },
            )

    def test_simple_addition_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_simple_addition_model()

    def test_simple_addition_model_failure(self):
        # Tests a case where the output does not match the expected output
        with self.assertRaises(ValueError):
            self.package_simple_addition_model(do_fail=True)


if __name__ == '__main__':
    unittest.main()
