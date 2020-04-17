#
# Uber, Inc. (c) 2018
#

import os
import tensorflow as tf
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropod.packagers import create_tensorflow_neuropod
from neuropod.tests.utils import get_string_concat_model_spec, check_strings_model


def create_tf_strings_model():
    """
    A model that concatenates two input strings
    """
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope("some_namespace"):
            x = tf.placeholder(tf.string, name="in_x")
            y = tf.placeholder(tf.string, name="in_y")

            # Assigned to a variable for clarity
            out = tf.string_join([x, y], separator=" ", name="out")  # noqa: F841

    return g.as_graph_def()


@unittest.skipIf(tf.__version__[0] == "2", "Skipping TF 1.x tests for TF 2.x")
class TestTensorflowStrings(unittest.TestCase):
    def package_strings_model(self, do_fail=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

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
                # Get the input/output spec along with test data
                **get_string_concat_model_spec(do_fail=do_fail)
            )

            # Run some additional checks
            check_strings_model(neuropod_path)

    def test_strings_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_strings_model()

    def test_strings_model_failure(self):
        # Tests a case where the output does not match the expected output
        with self.assertRaises(ValueError):
            self.package_strings_model(do_fail=True)


if __name__ == "__main__":
    unittest.main()
