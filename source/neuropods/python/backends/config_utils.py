#
# Uber, Inc. (c) 2018
#

import json
import os


def write_neuropod_config(neuropod_path, model_name, platform, input_spec, output_spec):
    """
    Creates the neuropod config file

    :param  neuropod_path:  The path to a neuropod package
    :param  model_name:     The name of the model (e.g. "my_addition_model")
    :param  platform:       The model type (e.g. "python", "pytorch", "tensorflow", etc.)

    :param  input_spec:     A list of dicts specifying the input to the model.
                            Ex: [{"name": "x", "dtype": "float32", "shape": (None, )}]

    :param  output_spec:    A list of dicts specifying the output of the model.
                            Ex: [{"name": "y", "dtype": "float32", "shape": (None, )}]
    """
    # TODO: Switch to prototext
    with open(os.path.join(neuropod_path, "config.json"), "w") as config_file:
        json.dump({
            "platform": platform,
            "input_spec": input_spec,
            "output_spec": output_spec,
        }, config_file)


def read_neuropod_config(neuropod_path):
    """
    Reads a neuropod config

    :param  neuropod_path:  The path to a neuropod package
    """
    with open(os.path.join(neuropod_path, "config.json"), "r") as config_file:
        return json.load(config_file)
