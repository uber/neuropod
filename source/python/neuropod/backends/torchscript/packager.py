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

import os
import shutil
import torch

from neuropod.utils.packaging_utils import packager


@packager(platform="torchscript")
def create_torchscript_neuropod(neuropod_path, module=None, module_path=None, **kwargs):
    """
    Packages a TorchScript model as a neuropod package.

    {common_doc_pre}

    :param  module:             An instance of a PyTorch ScriptModule. This model should return the outputs
                                as a dictionary. If this is not provided, `module_path` must be set.

                                !!! note ""
                                    For example, a model may output something like this:
                                    ```
                                    {
                                        "output1": value1,
                                        "output2": value2,
                                    }
                                    ```

    :param  module_path:        The path to a ScriptModule that was already exported using `torch.jit.save`.
                                If this is not provided, `module` must be set.

    {common_doc_post}
    """
    # Make sure the inputs are valid
    if (module is None) == (module_path is None):
        # If they are both None or both not None
        raise ValueError("Exactly one of 'module' and 'module_path' must be provided.")

    # Create a folder to store the model
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    os.makedirs(neuropod_data_path)

    # Add the model to the neuropod
    model_path = os.path.join(neuropod_data_path, "model.pt")
    if module_path is not None:
        # Copy in the module
        shutil.copyfile(module_path, model_path)
    else:
        # Save the model
        torch.jit.save(module, model_path)
