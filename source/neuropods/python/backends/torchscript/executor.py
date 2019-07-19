#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
from six import string_types
import torch

from neuropods.backends.neuropod_executor import NeuropodExecutor

SINGLE_OUTPUT_ERROR_MSG = "Please either return a dictionary from your model or provide an `output_spec` " \
                          "of size 1"

class TorchScriptNeuropodExecutor(NeuropodExecutor):
    """
    Executes a TorchScript neuropod
    """

    def __init__(self, neuropod_path, load_custom_ops=True):
        """
        Load a TorchScript neuropod

        :param  neuropod_path:  The path to a python neuropod package
        """
        super(TorchScriptNeuropodExecutor, self).__init__(neuropod_path)

        # Load custom ops (if any)
        if load_custom_ops and "custom_ops" in self.neuropod_config:
            for op in self.neuropod_config["custom_ops"]:
                torch.ops.load_library(os.path.join(neuropod_path, "0", "ops", op))

        # Load the model
        self.model = torch.jit.load(os.path.join(neuropod_path, "0", "data", "model.pt"))
        self.model_expects_dictionary = False

        # Check the expected input format of the model
        model_inputs = self.model.forward.schema.arguments

        if len(model_inputs) > 0 and model_inputs[0].type.kind() == "ClassType":
            model_inputs = model_inputs[1:]

        # Expects a dictionary mapping from a tensor name to tensor
        if len(model_inputs) == 1 and model_inputs[0].type.kind() == "DictType":
            self.model_expects_dictionary = True

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

        # Convert the inputs to torch tensors
        converted_inputs = {}
        for k, v in inputs.items():
            if v.dtype.type == np.str_:
                converted_inputs[k] = v.tolist()
            else:
                converted_inputs[k] = torch.from_numpy(v)

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

        # single output torch.Tensor or list<string>
        else:
            output_spec = self.neuropod_config["output_spec"]
            if not output_spec:
                raise RuntimeError("Output spec missing." + SINGLE_OUTPUT_ERROR_MSG)

            if len(output_spec) != 1:
                raise RuntimeError("Output spec has has more than one entry." + SINGLE_OUTPUT_ERROR_MSG)

            name = output_spec[0]["name"]
            dtype = output_spec[0]["dtype"]
            self._insert_value_to_output(neuropod_out, name, out, dtype=dtype)

        return neuropod_out

    def _insert_value_to_output(self, neuropod_out, key, value, dtype=None):
        if isinstance(value, torch.Tensor):
            neuropod_out[key] = value.cpu().numpy()
        elif isinstance(value, list) and (dtype == "string" or  isinstance(value[0], string_types)):
            neuropod_out[key] = np.array(value, dtype=np.str_)
        else:
            raise RuntimeError(
                "All outputs must be torch tensors or list of strings! Output `{}` was of type `{}`".format(
                    key,
                    type(value)))

        return neuropod_out
