#
# Uber, Inc. (c) 2018
#

import json
import numpy as np
import os

ALLOWED_DTYPES = [
    "float32",
    "float64",
    "string",

    "int8",
    "int16",
    "int32",
    "int64",

    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


def validate_tensor_spec(spec, no_device):
    """
    Validates a tensor spec

    :param  no_device:  If a device should be specifed or not
    """
    for item in spec:
        name = item["name"]
        dtype = item["dtype"]
        shape = item["shape"]


        if no_device:
            if "device" in spec:
                device = item["device"]
                raise ValueError("A device should not be included in this spec! Found '{}'".format(device))
        else:
            if "device" not in item:
                raise ValueError("A device was expected in this spec, but was not found.")

            device = item["device"]
            if device not in ["CPU", "GPU"]:
                raise ValueError("{} is not an allowed device. Must either be 'CPU' or 'GPU'.".format(device))

        if dtype not in ALLOWED_DTYPES:
            raise ValueError("{} is not an allowed data type!".format(dtype))

        if not isinstance(name, basestring):
            raise ValueError("Field 'name' must be a string! Got value {} of type {}.".format(name, type(name)))

        if not isinstance(shape, (list, tuple)):
            raise ValueError("Field 'shape' must be a tuple! Got value {} of type {}.".format(shape, type(shape)))

        for dim in shape:
            # A bool is an instance of an int so we have to do that check first
            is_uint = (not isinstance(dim, bool)) and isinstance(dim, (int, long)) and dim > 0

            if dim is None or is_uint or isinstance(dim, basestring):
                continue
            else:
                raise ValueError(
                    "All items in 'shape' must either be None, a string, or a positive integer! Got {}".format(dim))


def validate_neuropod_config(config):
    """
    Validates a neuropod config
    """
    name = config["name"]
    platform = config["platform"]

    if not isinstance(name, basestring):
        raise ValueError("Field 'name' in config must be a string! Got value {} of type {}.".format(name, type(name)))

    if not isinstance(platform, basestring):
        raise ValueError(
            "Field 'platform' in config must be a string! Got value {} of type {}.".format(
                platform, type(platform)))

    # Input specs should have devices
    validate_tensor_spec(config["input_spec"], no_device=False)

    # Output specs should not have devices
    validate_tensor_spec(config["output_spec"], no_device=True)


def canonicalize_tensor_spec(spec, default_device):
    """
    Converts the datatypes in a tensor spec to canonical versions
    (e.g. converts double to float64)

    :param  spec:               A list of tensor specifications
    :param  default_device:     The default device to use if one is not specified.
    """

    transformed = []
    for item in spec:
        canonical = {
            "name": item["name"],
            "dtype": np.dtype(item["dtype"]).name,
            "shape": item["shape"]
        }

        if default_device is not None:
            canonical["device"] = default_device

        if "device" in item:
            canonical["device"] = item["device"]

        transformed.append(canonical)
    return transformed


def write_neuropod_config(neuropod_path, model_name, platform, input_spec, output_spec, default_input_device=None):
    """
    Creates the neuropod config file

    :param  neuropod_path:  The path to a neuropod package
    :param  model_name:     The name of the model (e.g. "my_addition_model")
    :param  platform:       The model type (e.g. "python", "pytorch", "tensorflow", etc.)

    :param  input_spec:     A list of dicts specifying the input to the model. These can optionally specify
                            a device (either "CPU" or "GPU"). If not specified, the `default_input_device`
                            will be used.
                            Ex: [{"name": "x", "dtype": "float32", "shape": (None, ), "device": "GPU"}]

    :param  output_spec:    A list of dicts specifying the output of the model. These will always be moved to
                            CPU after inference.
                            Ex: [{"name": "y", "dtype": "float32", "shape": (None, )}]

    :param  default_input_device:   The default device that input tensors should be moved to before inference.
    """
    # TODO: Switch to prototext
    with open(os.path.join(neuropod_path, "config.json"), "w") as config_file:
        config = {
            "name": model_name,
            "platform": platform,
            "input_spec": canonicalize_tensor_spec(input_spec, default_device=default_input_device),

            # Outputs are always moved to CPU so we don't specify a device
            "output_spec": canonicalize_tensor_spec(output_spec, default_device=None),
        }

        # Verify that the config is correct
        validate_neuropod_config(config)

        # Write out the config as JSON
        json.dump(config, config_file, indent=4)


def read_neuropod_config(neuropod_path):
    """
    Reads a neuropod config

    :param  neuropod_path:  The path to a neuropod package
    """
    with open(os.path.join(neuropod_path, "config.json"), "r") as config_file:
        config = json.load(config_file)

        # Set devices on tensors without devices to ensure backwards compatibility
        config["input_spec"] = canonicalize_tensor_spec(config["input_spec"], default_device="GPU")

        # Outputs are always moved to CPU
        config["output_spec"] = canonicalize_tensor_spec(config["output_spec"], default_device=None)

        # Verify that the config is correct
        validate_neuropod_config(config)

        return config
