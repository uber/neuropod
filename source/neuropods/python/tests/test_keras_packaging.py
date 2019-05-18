#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropods.backends.keras.packager import create_keras_neuropod, \
    infer_keras_input_spec, infer_keras_output_spec
from neuropods.loader import load_neuropod
from neuropods.tests.utils import get_addition_model_spec, check_addition_model

def create_keras_addition_model():
    """
    A simple addition model
    """
    x = Input(batch_shape=(None,), name='x')
    y = Input(batch_shape=(None,), name='y')
    optional = Input(batch_shape=(None,), name='optional', dtype=tf.string)
    out = Add(name='out')([x, y])
    model = Model(inputs=[x, y, optional], outputs=[out])
    return model


class TestKerasPackaging(unittest.TestCase):
    def package_simple_addition_model(self, do_fail=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            # `create_keras_neuropod` runs inference with the test data immediately
            # after creating the neuropod. Raises a ValueError if the model output
            # does not match the expected output.
            create_keras_neuropod(
                neuropod_path=neuropod_path,
                model_name="addition_model",
                sess=tf.keras.backend.get_session(),
                model=create_keras_addition_model(),
                # Get the input/output spec along with test data
                **get_addition_model_spec(do_fail=do_fail)
            )

            # Run some additional checks
            check_addition_model(neuropod_path)

    def test_simple_addition_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_simple_addition_model()

    def test_simple_addition_model_failure(self):
        # Tests a case where the output does not match the expected output
        with self.assertRaises(ValueError):
            self.package_simple_addition_model(do_fail=True)

    def test_input_spec_inference(self):
        # Test whether addition model's input spec is inferred correctly
        inferred_spec = infer_keras_input_spec(create_keras_addition_model())
        self.assertEquals(get_addition_model_spec()['input_spec'], inferred_spec)

    def test_output_spec_inference(self):
        # Test whether addition model's output spec is inferred correctly
        inferred_spec = infer_keras_output_spec(create_keras_addition_model())
        self.assertEquals(get_addition_model_spec()['output_spec'], inferred_spec)


if __name__ == '__main__':
    unittest.main()
