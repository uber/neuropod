#
# Uber, Inc. (c) 2018
#

import os
import torch

from neuropods.backends import config_utils
from neuropods.utils.eval_utils import load_and_test_neuropod


def create_torchscript_neuropod(
        neuropod_path,
        model_name,
        module,
        input_spec,
        output_spec,
        test_input_data=None,
        test_expected_out=None):
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
    """
    try:
        # Create the neuropod folder
        os.mkdir(neuropod_path)
    except OSError:
        raise ValueError("The specified neuropod path ({}) already exists! Aborting...".format(neuropod_path))

    # Write the neuropod config file
    config_utils.write_neuropod_config(
        neuropod_path=neuropod_path,
        model_name=model_name,
        platform="torchscript",
        input_spec=input_spec,
        output_spec=output_spec,
    )

    # Create a folder to store the model
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    os.makedirs(neuropod_data_path)

    # Save the model
    model_path = os.path.join(neuropod_data_path, "model.pt")
    torch.jit.save(module, model_path)

    if test_input_data is not None:
        # Load and run the neuropod to make sure that packaging worked correctly
        # Throws a ValueError if the output doesn't match the expected output (if specified)
        load_and_test_neuropod(
            neuropod_path,
            test_input_data,
            test_expected_out,
        )
