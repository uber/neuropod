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

import numpy as np
import os
import torch
import unittest
from testpath.tempdir import TemporaryDirectory
from torch import Tensor
from typing import Dict

from neuropod.packagers import create_torchscript_neuropod
from neuropod.utils.eval_utils import load_and_test_neuropod
from neuropod.tests.utils import requires_frameworks


class DevicesModel(torch.jit.ScriptModule):
    """
    This model returns which device its inputs are on (0 for cpu and 1 for gpu)
    """

    @torch.jit.script_method
    def get_device(self, item):
        if item.device == torch.device("cpu"):
            return torch.tensor([0])
        else:
            return torch.tensor([1])

    @torch.jit.script_method
    def forward(self, data):
        # type: (Dict[str, Tensor])

        # This doesn't work from C++ because of a libtorch bug
        # out = {}
        #
        # for k in data.keys():
        #    if data[k].device == torch.device("cpu"):
        #        out[k] = torch.tensor([0])
        #    else:
        #        out[k] = torch.tensor([1])
        #
        # return out

        return {
            "x": self.get_device(data["x"]),
            "y": self.get_device(data["y"]),
        }


@requires_frameworks("torchscript")
class TestTorchScriptDevices(unittest.TestCase):
    def package_devices_model(self):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            # `create_torchscript_neuropod` runs inference with the test data immediately
            # after creating the neuropod. Raises a ValueError if the model output
            # does not match the expected output.
            create_torchscript_neuropod(
                neuropod_path=neuropod_path,
                model_name="devices_model",
                module=DevicesModel(),
                input_spec=[
                    {"name": "x", "dtype": "float32", "shape": (None,)},
                    {"name": "y", "dtype": "float32", "shape": (None,)},
                ],
                output_spec=[
                    {"name": "x", "dtype": "int64", "shape": (None,)},
                    {"name": "y", "dtype": "int64", "shape": (None,)},
                ],
                test_input_data={
                    "x": np.arange(5).astype(np.float32),
                    "y": np.arange(5).astype(np.float32),
                },
                test_expected_out={
                    "x": np.array([0], dtype=np.int64),
                    "y": np.array([1], dtype=np.int64),
                },
                input_tensor_device={"x": "CPU"},
                default_input_tensor_device="GPU",
            )

            # Ensure all inputs are moved to CPU if we run with no visible GPUs
            load_and_test_neuropod(
                neuropod_path,
                test_input_data={
                    "x": np.arange(5).astype(np.float32),
                    "y": np.arange(5).astype(np.float32),
                },
                test_expected_out={
                    "x": np.array([0], dtype=np.int64),
                    "y": np.array([0], dtype=np.int64),
                },
                neuropod_load_args={"visible_gpu": None},
            )

    def test_device_model(self):
        # Make sure that packaging works correctly and
        # the tensors are on the correct devices
        self.package_devices_model()


if __name__ == "__main__":
    unittest.main()
