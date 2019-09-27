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

from neuropods.backends.tensorflow.trt import is_trt_available
from neuropods.backends.keras.packager import create_keras_neuropod, \
    infer_keras_input_spec, infer_keras_output_spec
from neuropods.loader import load_neuropod
from neuropods.tests.utils import get_addition_model_spec, check_addition_model

def create_keras_addition_model(node_name_mapping=None):
    """
    A simple addition model
    """
    if node_name_mapping is None:
        class id_dict(dict):
            def __missing__(self, key):
                return key

        node_name_mapping = id_dict()

    x = Input(batch_shape=(None,), name=node_name_mapping['x'])
    y = Input(batch_shape=(None,), name=node_name_mapping['y'])
    optional = Input(batch_shape=(None,), name=node_name_mapping['optional'], dtype=tf.string)
    out = Add(name=node_name_mapping['out'])([x, y])
    model = Model(inputs=[x, y, optional], outputs=[out])
    return model


class TestKerasPackaging(unittest.TestCase):
    def package_simple_addition_model(self, alias_names=False, do_fail=False, use_trt=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            if alias_names:
                node_name_mapping = {'x': 'x_', 'y': 'y_', 'optional': 'optional_', 'out': 'out_'}
            else:
                node_name_mapping = None

            # `create_keras_neuropod` runs inference with the test data immediately
            # after creating the neuropod. Raises a ValueError if the model output
            # does not match the expected output.
            create_keras_neuropod(
                neuropod_path=neuropod_path,
                model_name="addition_model",
                sess=tf.keras.backend.get_session(),
                model=create_keras_addition_model(node_name_mapping),
                node_name_mapping=node_name_mapping,
                use_trt=use_trt,
                # Get the input/output spec along with test data
                **get_addition_model_spec(do_fail=do_fail)
            )

            # Run some additional checks
            check_addition_model(neuropod_path)

    def test_simple_addition_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_simple_addition_model()

    def test_simple_addition_model_with_alias(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_simple_addition_model(alias_names=True)

    @unittest.skipIf(not is_trt_available(), "TRT is not available in this version of TF")
    def test_simple_addition_model_trt(self):
        # Tests TRT optimization
        self.package_simple_addition_model(use_trt=True)

    @unittest.skipIf(not is_trt_available(), "TRT is not available in this version of TF")
    def test_simple_addition_model_with_alias_trt(self):
        # Tests TRT optimization
        self.package_simple_addition_model(alias_names=True, use_trt=True)

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
