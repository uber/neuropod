#
# Uber, Inc. (c) 2018
#

import os
import torch

from neuropods.utils.packaging_utils import create_neuropod

def create_torchscript_neuropod(
        module,
        **kwargs
        ):
    """
    Packages a TorchScript model as a neuropod package.

    :param  neuropod_path:      The output neuropod path

    :param  model_name:         The name of the model

    :param  module:             An instance of a PyTorch ScriptModule. This model should return the outputs
                                as a dictionary. For example, a model may output something like this:
                                    {
                                        "output1": value1,
                                        "output2": value2,
                                    }

    :param  input_spec:         A list of dicts specifying the input to the model. For each input, if shape
                                is set to `None`, no validation is done on the shape. If shape is a tuple, the
                                dimensions of the input are validated against that tuple.  A value of
                                `None` for any of the dimensions means that dimension will not be checked.
                                `dtype` can be any valid numpy datatype string.
                                Ex: [
                                    {"name": "x", "dtype": "float32", "shape": (None,)},
                                    {"name": "y", "dtype": "float32", "shape": (None,)},
                                ]

    :param  output_spec:        A list of dicts specifying the output of the model. See the documentation for
                                the `input_spec` parameter for more details.
                                Ex: [
                                    {"name": "out", "dtype": "float32", "shape": (None,)},
                                ]

    :param  test_input_data:    Optional sample input data. This is a dict mapping input names to
                                values. If this is provided, inference will be run in an isolated environment
                                immediately after packaging to ensure that the neuropod was created
                                successfully. Must be provided if `test_expected_out` is provided.

                                Throws a ValueError if inference failed.
                                Ex: {
                                    "x": np.arange(5),
                                    "y": np.arange(5),
                                }

    :param  test_expected_out:  Optional expected output. Throws a ValueError if the output of model inference
                                does not match the expected output.
                                Ex: {
                                    "out": np.arange(5) + np.arange(5)
                                }

    :param  persist_test_data:  Optionally saves the test data within the packaged neuropod. default True.

    :param  input_tensor_device:    A dict mapping input tensor names to the device
                                    that the model expects them to be on. This can
                                    either be `GPU` or `CPU`. Any tensors in `input_spec`
                                    not specified in this mapping will use the
                                    `default_input_tensor_device` specified below.
                                    If a GPU is selected at inference time, Neuropods
                                    will move tensors to the appropriate devices before
                                    running the model. Otherwise, it will attempt to run
                                    the model on CPU and move all tensors (and the model)
                                    to CPU.
                                    See the docstring for `load_neuropod` for more info.
                                    Ex: `{"x": "GPU"}`

    :param  default_input_tensor_device:    The default device that input tensors are expected
                                            to be on. This can either be `GPU` or `CPU`.
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
