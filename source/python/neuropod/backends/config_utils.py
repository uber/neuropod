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

import json
import os
from six import string_types, integer_types

from neuropod.utils.dtype_utils import get_dtype_name

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


def validate_tensor_spec(spec):
    """
    Validates a tensor spec
    """
    for item in spec:
        name = item["name"]
        dtype = item["dtype"]
        shape = item["shape"]

        if dtype not in ALLOWED_DTYPES:
            raise ValueError("{} is not an allowed data type!".format(dtype))

        if not isinstance(name, string_types):
            raise ValueError(
                "Field 'name' must be a string! Got value {} of type {}.".format(
                    name, type(name)
                )
            )

        if not isinstance(shape, (list, tuple)):
            raise ValueError(
                "Field 'shape' must be a tuple! Got value {} of type {}.".format(
                    shape, type(shape)
                )
            )

        for dim in shape:
            # A bool is an instance of an int so we have to do that check first
            is_uint = (
                (not isinstance(dim, bool))
                and isinstance(dim, integer_types)
                and dim > 0
            )

            if dim is None or is_uint or isinstance(dim, string_types):
                continue
            else:
                raise ValueError(
                    "All items in 'shape' must either be None, a string, or a positive integer! Got {}".format(
                        dim
                    )
                )


def validate_neuropod_config(config):
    """
    Validates a neuropod config
    """
    name = config["name"]
    platform = config["platform"]
    device_mapping = config["input_tensor_device"]

    if not isinstance(name, string_types):
        raise ValueError(
            "Field 'name' in config must be a string! Got value {} of type {}.".format(
                name, type(name)
            )
        )

    if not isinstance(platform, string_types):
        raise ValueError(
            "Field 'platform' in config must be a string! Got value {} of type {}.".format(
                platform, type(platform)
            )
        )

    validate_tensor_spec(config["input_spec"])
    validate_tensor_spec(config["output_spec"])

    # Optional custom ops
    if "custom_ops" in config:
        custom_ops = config["custom_ops"]

        if not isinstance(custom_ops, list):
            raise ValueError(
                "Optional field 'custom_ops' must be a list! Got value {} of type {}".format(
                    custom_ops, type(custom_ops)
                )
            )

        for op in custom_ops:
            if not isinstance(op, string_types):
                raise ValueError(
                    "All items in 'custom_ops' must be strings! Got value {} of type {}.".format(
                        op, type(op)
                    )
                )

    # Ensure all inputs have a device specified
    input_tensor_names = {item["name"] for item in config["input_spec"]}
    device_tensor_names = set(device_mapping.keys())
    inputs_without_device = input_tensor_names - device_tensor_names
    devices_without_input = device_tensor_names - input_tensor_names

    if len(inputs_without_device) != 0:
        raise ValueError(
            "Some input tensors do not have devices specified: {}".format(
                inputs_without_device
            )
        )

    if len(devices_without_input) != 0:
        raise ValueError(
            "Devices were specified for some tensors not in the `input_spec`: {}".format(
                devices_without_input
            )
        )

    for tensor_name, device in device_mapping.items():
        if device not in ["GPU", "CPU"]:
            raise ValueError(
                "Device must either be 'GPU' or 'CPU'! Got value '{}' for tensor named '{}'.".format(
                    device, tensor_name
                )
            )


def canonicalize_tensor_spec(spec):
    """
    Converts the datatypes in a tensor spec to canonical versions
    (e.g. converts double to float64)
    """
    transformed = []
    for item in spec:
        transformed.append(
            {
                "name": item["name"],
                "dtype": get_dtype_name(item["dtype"]),
                "shape": item["shape"],
            }
        )
    return transformed


def write_neuropod_config(
    neuropod_path,
    model_name,
    platform,
    input_spec,
    output_spec,
    platform_version_semver="*",
    custom_ops=None,
    input_tensor_device=None,
    default_input_tensor_device="GPU",
    **kwargs
):
    """
    Creates the neuropod config file

    :param  neuropod_path:  The path to a neuropod package
    :param  model_name:     The name of the model (e.g. "my_addition_model")
    :param  platform:       The model type (e.g. "python", "pytorch", "tensorflow", etc.)

    :param  platform_version_semver: The required platform version specified as semver range
                                     See https://semver.org/, https://docs.npmjs.com/misc/semver#ranges
                                     or https://docs.npmjs.com/misc/semver#advanced-range-syntax for
                                     examples and more info. Default is `*` (any version is okay)
                                     Ex: `1.13.1` or `> 1.13.1`

    :param  input_spec:     A list of dicts specifying the input to the model.
                            Ex: [{"name": "x", "dtype": "float32", "shape": (None, )}]

    :param  output_spec:    A list of dicts specifying the output of the model.
                            Ex: [{"name": "y", "dtype": "float32", "shape": (None, )}]

    :param  input_tensor_device:    A dict mapping input tensor names to the device
                                    that the model expects them to be on. This can
                                    either be `GPU` or `CPU`. Any tensors in `input_spec`
                                    not specified in this mapping will use the
                                    `default_input_tensor_device` specified below.

                                    If a GPU is selected at inference time, Neuropod
                                    will move tensors to the appropriate devices before
                                    running the model. Otherwise, it will attempt to run
                                    the model on CPU and move all tensors (and the model)
                                    to CPU.

                                    See the docstring for `load_neuropod` for more info.

                                    Ex: `{"x": "GPU"}`

    :param  default_input_tensor_device:    The default device that input tensors are expected
                                            to be on. This can either be `GPU` or `CPU`.

    """
    if custom_ops is None:
        custom_ops = []

    if input_tensor_device is None:
        input_tensor_device = {}

    # Canonicalize the specs
    input_spec = canonicalize_tensor_spec(input_spec)
    output_spec = canonicalize_tensor_spec(output_spec)

    # Set up the device mapping
    device_mapping = {}
    for item in input_spec:
        name = item["name"]
        if name in input_tensor_device:
            # Use the device specified by the user
            device_mapping[name] = input_tensor_device[name]
        else:
            # Use the default device
            device_mapping[name] = default_input_tensor_device

    # TODO: Switch to prototext
    with open(os.path.join(neuropod_path, "config.json"), "w") as config_file:
        config = {
            "name": model_name,
            "platform": platform,
            "platform_version_semver": platform_version_semver,
            "input_spec": input_spec,
            "output_spec": output_spec,
            "custom_ops": custom_ops,
            "input_tensor_device": device_mapping,
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

        # For backwards compatibility
        # TODO(vip): Remove this on version increase
        if "input_tensor_device" not in config:
            # If there is no mapping in the configuration, move all tensors to
            # GPU by default
            config["input_tensor_device"] = {
                item["name"]: "GPU" for item in config["input_spec"]
            }

        # Verify that the config is correct
        validate_neuropod_config(config)

        return config
