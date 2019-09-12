#
# Uber, Inc. (c) 2019
#

import unittest

# Make sure that all frameworks report that we don't have GPUs
class TestGPUEnvSupport(unittest.TestCase):
    def test_torch_gpu(self):
        import torch
        self.assertFalse(torch.cuda.is_available())

    def test_tf_gpu(self):
        import tensorflow as tf
        self.assertFalse(tf.test.is_built_with_cuda() and tf.test.is_gpu_available())

    def test_neuropod_is_cuda_available(self):
        import neuropods_native
        self.assertFalse(neuropods_native.is_cuda_available())


if __name__ == '__main__':
    unittest.main()
