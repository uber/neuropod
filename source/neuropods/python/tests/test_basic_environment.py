#
# Uber, Inc. (c) 2019
#

import unittest

class TestBasicEnv(unittest.TestCase):
    def test_tf_trt_available(self):
        # Make sure we correctly report TRT availability
        import tensorflow as tf
        from neuropods.backends.tensorflow.trt import is_trt_available

        # TODO(vip): min version check
        if tf.__version__ == '1.14.0':
            self.assertTrue(is_trt_available())
        else:
            self.assertFalse(is_trt_available())


if __name__ == '__main__':
    unittest.main()
