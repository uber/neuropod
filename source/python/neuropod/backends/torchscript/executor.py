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
from six import string_types
import torch

from neuropod.backends.neuropod_executor import NeuropodExecutor
from neuropod.utils.hash_utils import sha256sum

SINGLE_OUTPUT_ERROR_MSG = (
    "Please either return a dictionary from your model or provide an `output_spec` "
    "of size 1"
)


# Avoid loading the same custom op twice
loaded_op_hashes = set()


def isnamedtuple(x):
    """
    Utility to check whether something is a named tuple
    Since a namedtuple is basically a tuple with some additional metadata, we can't just do an `isinstance` check
    Based on https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple/2166841#2166841
    """
    t = type(x)
    b = t.__bases__

    # Named tuples are tuple subclasses
    if len(b) != 1 or b[0] != tuple:
        return False

    # They have a fields tuple
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False

    # All the items in the fields tuple are strings
    return all(type(n) == str for n in f)


class TorchScriptNeuropodExecutor(NeuropodExecutor):
    """
    Executes a TorchScript neuropod
    """

    def __init__(self, neuropod_path, visible_gpu=0, load_custom_ops=True):
        """
        Load a TorchScript neuropod

        :param  neuropod_path:      The path to a TorchScript neuropod package
        :param  visible_gpu:        The index of the GPU that this Neuropod should run on (if any).
                                    This is either `None` or a nonnegative integer. Setting this
                                    to `None` will attempt to run this model on CPU.
        :param  load_custom_ops:    Whether or not to load custom ops included in the model.
        """
        super(TorchScriptNeuropodExecutor, self).__init__(neuropod_path)
        self.visible_gpu = visible_gpu

        # Load custom ops (if any)
        if load_custom_ops and "custom_ops" in self.neuropod_config:
            for op in self.neuropod_config["custom_ops"]:
                lib_path = os.path.join(neuropod_path, "0", "ops", op)
                lib_hash = sha256sum(lib_path)
                if lib_hash not in loaded_op_hashes:
                    torch.ops.load_library(lib_path)
                    loaded_op_hashes.add(lib_hash)

        # Load the model onto the appropriate device (ideally a GPU if we have one available)
        self.model = torch.jit.load(
            os.path.join(neuropod_path, "0", "data", "model.pt"),
            map_location=self._get_torch_device("GPU"),
        )
        self.model_expects_dictionary = False

        # Check the expected input format of the model
        model_inputs = self.model.forward.schema.arguments

        if len(model_inputs) > 0 and model_inputs[0].type.kind() == "ClassType":
            model_inputs = model_inputs[1:]

        # Expects a dictionary mapping from a tensor name to tensor
        if len(model_inputs) == 1 and model_inputs[0].type.kind() == "DictType":
            self.model_expects_dictionary = True

    def _get_torch_device(self, target_device):
        """
        Get a concrete device (e.g. `cuda:0` or `cpu`) given a target (e.g. `CPU` or `GPU`)
        """
        if self.visible_gpu is None or not torch.cuda.is_available():
            # No matter what the target device is, we don't have a choice other
            # than running on CPU
            # TODO(vip): warn if visible_gpu is set but CUDA isn't available
            return "cpu"

        if target_device == "CPU":
            return "cpu"
        elif target_device == "GPU":
            return "cuda:" + str(self.visible_gpu)

        raise ValueError("Invalid device '{}'!".format(target_device))

    def forward(self, inputs):
        """
        Run inference using the specifed inputs.

        :param  inputs:     A dict mapping input names to values. This must match the input
                            spec in the neuropod config for the loaded model.
                            Ex: {'x1': np.array([5]), 'x2': np.array([6])}
                            *Note:* all the keys in this dict must be strings and all the
                            values must be numpy arrays

        :returns:   A dict mapping output names to values. All the keys
                    in this dict are strings and all the values are numpy arrays.
        """

        # Convert the inputs to torch tensors and move to the appropriate device
        converted_inputs = {}
        for k, v in inputs.items():
            # Get the target device for this tensor
            target_device = self._get_torch_device(self.input_device_mapping[k])

            if v.dtype.type == np.str_:
                converted_inputs[k] = v.tolist()

                # We don't handle devices for string "tensors" because lists cannot
                # be moved to GPU
                # TODO(vip): warn if target_device.startswith("cuda")
            else:
                converted_inputs[k] = torch.from_numpy(v)

                # Move to the correct device
                converted_inputs[k] = converted_inputs[k].to(target_device)

        # Run inference
        with torch.no_grad():
            if self.model_expects_dictionary:
                out = self.model(converted_inputs)
            else:
                out = self.model(**converted_inputs)

        neuropod_out = {}
        if isinstance(out, dict):
            # Convert the outputs to numpy arrays
            # acceptable values must be torch.Tensor or lists of strings
            for key, value in out.items():
                neuropod_out = self._insert_value_to_output(neuropod_out, key, value)

        elif isnamedtuple(out):
            # This is a named tuple
            for key, value in out._asdict().items():
                neuropod_out = self._insert_value_to_output(neuropod_out, key, value)

        elif isinstance(out, tuple):
            # Each item in this tuple should be a dict
            for d in out:
                if isinstance(d, dict):
                    # Convert the outputs to numpy arrays
                    # acceptable values must be torch.Tensor or lists of strings
                    for key, value in d.items():
                        neuropod_out = self._insert_value_to_output(
                            neuropod_out, key, value
                        )
                else:
                    raise RuntimeError(
                        "When returning a tuple, each item must be a dict. Got item of type {}".format(
                            type(value)
                        )
                    )
        else:
            # single output torch.Tensor or list<string>
            output_spec = self.neuropod_config["output_spec"]
            if not output_spec:
                raise RuntimeError("Output spec missing." + SINGLE_OUTPUT_ERROR_MSG)

            if len(output_spec) != 1:
                raise RuntimeError(
                    "Output spec has has more than one entry." + SINGLE_OUTPUT_ERROR_MSG
                )

            name = output_spec[0]["name"]
            dtype = output_spec[0]["dtype"]
            self._insert_value_to_output(neuropod_out, name, out, dtype=dtype)

        return neuropod_out

    def _insert_value_to_output(self, neuropod_out, key, value, dtype=None):
        if key in neuropod_out:
            raise RuntimeError(
                "An item with name `{}` was already returned by this model. Please ensure your model does not have duplicate outputs".format(
                    key
                )
            )

        if isinstance(value, torch.Tensor):
            neuropod_out[key] = value.cpu().numpy()
        elif isinstance(value, list) and (
            dtype == "string" or isinstance(value[0], string_types)
        ):
            neuropod_out[key] = np.array(value, dtype=np.str_)
        else:
            raise RuntimeError(
                "All outputs must be torch tensors or list of strings! Output `{}` was of type `{}`".format(
                    key, type(value)
                )
            )

        return neuropod_out
