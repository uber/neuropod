#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import subprocess
import sys
import tensorflow as tf
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropods.packagers import create_tensorflow_neuropod
from neuropods.loader import load_neuropod
from neuropods.tests.utils import get_addition_model_spec

def create_tf_addition_model(custom_op_path):
    """
    A simple addition model that uses a custom op
    """
    addition_op_module = tf.load_op_library(custom_op_path)

    g = tf.Graph()
    with g.as_default():
        with tf.name_scope("some_namespace"):
            x = tf.placeholder(tf.float32, name="in_x")
            y = tf.placeholder(tf.float32, name="in_y")

            out = addition_op_module.neuropod_addition(x, y, name="out")

    return g.as_graph_def()

@unittest.skipIf(tf.__version__ == '1.14.0' and sys.platform == 'darwin',
                 'See https://github.com/tensorflow/tensorflow/issues/30633')
class TestTensorflowCustomOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the custom op
        # TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
        # TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
        # g++ -std=c++11 -shared addition_op.cc -o addition_op.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
        current_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.check_call([os.getenv("TF_CXX", "g++"), "-std=c++11", "-shared", "addition_op.cc", "-o", "addition_op.so", "-fPIC"] + tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags() + ["-O2"], cwd=current_dir)
        cls.custom_op_path = os.path.join(current_dir, "addition_op.so")

    def package_simple_addition_model(self, do_fail=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            # `create_tensorflow_neuropod` runs inference with the test data immediately
            # after creating the neuropod. Raises a ValueError if the model output
            # does not match the expected output.
            create_tensorflow_neuropod(
                neuropod_path=neuropod_path,
                model_name="addition_model",
                graph_def=create_tf_addition_model(self.custom_op_path),
                node_name_mapping={
                    "x": "some_namespace/in_x:0",
                    "y": "some_namespace/in_y:0",
                    "out": "some_namespace/out:0",
                },
                custom_ops=[self.custom_op_path],
                # Get the input/output spec along with test data
                **get_addition_model_spec(do_fail=do_fail)
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
