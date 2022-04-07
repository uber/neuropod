# Copyright (c) 2022 The Neuropod Authors
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

import os
import unittest
import torch

import numpy as np

from distutils.version import LooseVersion
from typing import Dict, List, Union
from testpath.tempdir import TemporaryDirectory

from neuropod.packagers import create_torchscript_neuropod
from neuropod.tests.utils import requires_frameworks


class Model(torch.nn.Module):
    def __init__(self, vocab: Dict[str, int]):
        super().__init__()
        self.vocab: Dict[str, int] = vocab

    def preprocess(
        self, inputs: Union[List[str], torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        # the following check is to help jit compiler know the type of inputs is List[str]
        if isinstance(inputs, torch.Tensor):
            raise ValueError("unsupported input type")
        indexed_inputs: List[int] = []
        for val in inputs:
            indexed_inputs.append(self.vocab.get(val, 0))
        return torch.tensor(indexed_inputs, device=device).reshape(
            [len(indexed_inputs), 1]
        )

    def forward(
        self, inputs: Dict[str, Union[List[str], torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        num_col = inputs["b"]
        # assertion is used to help compiler know that num_col is Tensor type
        assert isinstance(num_col, torch.Tensor)
        str_col = inputs["a"]
        return {
            "c": torch.cat([self.preprocess(str_col, num_col.device), num_col], dim=-1)
        }


@unittest.skipIf(
    LooseVersion(torch.__version__) < LooseVersion("1.10.0"),
    "Torch supports UnionType after 1.10, skip the test if torch version < 1.10",
)
@requires_frameworks("torchscript")
class TestTorchScriptDictWithUnionValueTypeModel(unittest.TestCase):
    def test_dict_with_union_value_type_model(self):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            # `create_torchscript_neuropod` runs inference with the test data immediately
            # after creating the neuropod. Raises a ValueError if the model output
            # does not match the expected output.
            create_torchscript_neuropod(
                neuropod_path=neuropod_path,
                model_name="dict_with_union_value_type_model",
                module=torch.jit.script(Model(vocab={"a": 1, "b": 2})),
                input_spec=[
                    {"name": "a", "dtype": "string", "shape": (None,)},
                    {"name": "b", "dtype": "float32", "shape": (None, 1)},
                ],
                output_spec=[{"name": "c", "dtype": "float32", "shape": (None, 2)},],
                test_input_data={
                    "a": np.array(["a", "b", "c"]),
                    "b": np.array([[1], [2], [3]], dtype=np.float32),
                },
                test_expected_out={
                    "c": np.array([[1, 1], [2, 2], [0, 3]], dtype=np.float32),
                },
            )
