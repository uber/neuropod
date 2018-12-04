#
# Uber, Inc. (c) 2018
#

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
        super(TorchScriptNeuropodExecutor, self).__init__()

        # Load the model
        self.model = torch.jit.load(os.path.join(neuropod_path, "0", "data", "model.pt"))

    def forward(self, inputs):
        """
        Run inference using the specifed inputs.

        :param  inputs:     A dict mapping input names to values. This must match the input
                            spec in the neuropod config for the loaded model.
                            Ex: {'x1': [5], 'x2': [6]}

        :returns:   A dict mapping output names to values. This is checked to ensure that it
                    matches the spec in the neuropod config for the loaded model.
        """

        # Convert the inputs to torch tensors
        converted_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            out = self.model(**converted_inputs)

        # Make sure we have a tuple of tuples
        if out and not isinstance(out[0], tuple):
            out = (out, )

        # Convert the outputs to numpy arrays
        converted_out = {}
        for key, value in out:
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    "All outputs must be torch tensors! Output `{}` was of type `{}`".format(
                        key,
                        type(value)))

            converted_out[key] = value.numpy()

        return converted_out
