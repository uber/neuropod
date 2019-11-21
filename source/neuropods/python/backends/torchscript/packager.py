#
# Uber, Inc. (c) 2018
#

import os
import torch

from neuropods.utils.packaging_utils import packager


@packager(platform="torchscript")
def create_torchscript_neuropod(neuropod_path, module, **kwargs):
    """
    Packages a TorchScript model as a neuropod package.

    {common_doc_pre}

    :param  module:             An instance of a PyTorch ScriptModule. This model should return the outputs
                                as a dictionary.

                                !!! note ""
                                    For example, a model may output something like this:
                                    ```
                                    {
                                        "output1": value1,
                                        "output2": value2,
                                    }
                                    ```

    {common_doc_post}
    """
    # Create a folder to store the model
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    os.makedirs(neuropod_data_path)

    # Save the model
    model_path = os.path.join(neuropod_data_path, "model.pt")
    torch.jit.save(module, model_path)
