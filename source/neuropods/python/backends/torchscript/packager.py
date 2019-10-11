#
# Uber, Inc. (c) 2018
#

import os
import torch

from neuropods.utils.packaging_utils import create_neuropod, set_packager_docstring

@set_packager_docstring
def create_torchscript_neuropod(
        module,
        **kwargs
        ):
    """
    Packages a TorchScript model as a neuropod package.

    {common_doc_pre}

    :param  module:             An instance of a PyTorch ScriptModule. This model should return the outputs
                                as a dictionary. For example, a model may output something like this:
                                    {
                                        "output1": value1,
                                        "output2": value2,
                                    }

    {common_doc_post}
    """
    def packager_fn(neuropod_path):
        # Create a folder to store the model
        neuropod_data_path = os.path.join(neuropod_path, "0", "data")
        os.makedirs(neuropod_data_path)

        # Save the model
        model_path = os.path.join(neuropod_data_path, "model.pt")
        torch.jit.save(module, model_path)

    create_neuropod(
        packager_fn=packager_fn,
        platform="torchscript",
        **kwargs
    )
