#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import tensorflow as tf
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropods.backends.tensorflow.packager import create_tensorflow_neuropod


def create_tf_strings_model():
    """
    A model that concatenates two input strings
    """
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope("some_namespace"):
            x = tf.placeholder(tf.string, name="in_x")
            y = tf.placeholder(tf.string, name="in_y")

            # UATG(flake8/F841) Assigned to a variable for clarity
            out = tf.string_join([x, y], separator=" ", name="out")

    return g.as_graph_def()


class TestTensorflowStrings(unittest.TestCase):
    def package_strings_model(self, do_fail=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            if do_fail:
                expected_out = np.array(["a", "b", "c"])
            else:
                expected_out = np.array(["apple sauce", "banana pudding", "carrot cake"])

            # `create_tensorflow_neuropod` runs inference with the test data immediately
            # after creating the neuropod. Raises a ValueError if the model output
            # does not match the expected output.
            create_tensorflow_neuropod(
                neuropod_path=neuropod_path,
                model_name="strings_model",
                graph_def=create_tf_strings_model(),
                node_name_mapping={
                    "x": "some_namespace/in_x:0",
                    "y": "some_namespace/in_y:0",
                    "out": "some_namespace/out:0",
                },
                input_spec=[
                    {"name": "x", "dtype": "string", "shape": (None,)},
                    {"name": "y", "dtype": "string", "shape": (None,)},
                ],
                output_spec=[
                    {"name": "out", "dtype": "string", "shape": (None,)},
                ],
                test_input_data={
                    "x": np.array(["apple", "banana", "carrot"]),
                    "y": np.array(["sauce", "pudding", "cake"]),
                },
                test_expected_out={
                    "out": expected_out,
                },
            )

    def test_strings_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_strings_model()

    def test_strings_model_failure(self):
        # Tests a case where the output does not match the expected output
        with self.assertRaises(ValueError):
            self.package_strings_model(do_fail=True)


if __name__ == '__main__':
    unittest.main()
