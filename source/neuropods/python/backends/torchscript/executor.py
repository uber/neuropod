#
# Uber, Inc. (c) 2018
#

import numpy as np
import os
import torch

from neuropods.backends.neuropod_executor import NeuropodExecutor


class TorchScriptNeuropodExecutor(NeuropodExecutor):
    """
    Executes a TorchScript neuropod
    """

    def __init__(self, neuropod_path):
        """
        Load a TorchScript neuropod

        :param  neuropod_path:  The path to a python neuropod package
        """
        super(TorchScriptNeuropodExecutor, self).__init__(neuropod_path)

        # Load the model
        self.model = torch.jit.load(os.path.join(neuropod_path, "0", "data", "model.pt"))

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
            if v.dtype.type == np.string_:
                converted_inputs[k] = v.tolist()
            else:
                converted_inputs[k] = torch.from_numpy(v)

        # Run inference
        with torch.no_grad():
            out = self.model(**converted_inputs)

        # Convert the outputs to numpy arrays
        converted_out = {}
        for key, value in out.items():
            if isinstance(value, torch.Tensor):
                converted_out[key] = value.numpy()
            elif isinstance(value, list) and isinstance(value[0], basestring):
                converted_out[key] = np.array(value, dtype=np.string_)
            else:
                raise RuntimeError(
                    "All outputs must be torch tensors! Output `{}` was of type `{}`".format(
                        key,
                        type(value)))

        return converted_out
