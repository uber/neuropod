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

import unittest


from neuropod.tests.utils import requires_frameworks


class TestGPUEnvSupport(unittest.TestCase):
    """
    Tests that the environment running the tests has GPUs available
    and that various frameworks used in the tests support using the
    GPUs
    """

    @requires_frameworks("torchscript")  # TODO(vip): Maybe find a better way to do this
    def test_torch_gpu(self):
        import torch

        self.assertTrue(torch.cuda.is_available())

    @requires_frameworks("tensorflow")
    def test_tf_gpu(self):
        import tensorflow as tf

        self.assertTrue(tf.test.is_built_with_cuda() and tf.test.is_gpu_available())


if __name__ == "__main__":
    unittest.main()
