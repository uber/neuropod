#
# Uber, Inc. (c) 2019
#

import unittest

# Tests that the environment running the tests has GPUs available
# and that various frameworks used in the tests support using the
# GPUs
class TestGPUEnvSupport(unittest.TestCase):
    def test_torch_gpu(self):
        import torch
        self.assertTrue(torch.cuda.is_available())

    def test_tf_gpu(self):
        import tensorflow as tf
        self.assertTrue(tf.test.is_built_with_cuda() and tf.test.is_gpu_available())

    def test_neuropod_is_cuda_available(self):
        import neuropods_native
        self.assertTrue(neuropods_native.is_cuda_available())


if __name__ == '__main__':
    unittest.main()
