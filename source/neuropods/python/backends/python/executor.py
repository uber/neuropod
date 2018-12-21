#
# Uber, Inc. (c) 2018
#

import os
import sys
import importlib
import json

import numpy as np

from neuropods.backends.neuropod_executor import NeuropodExecutor


def _validate_not_importable(package):
    """
    Try to import `package` and all its parents and verify that they are not
    importable. Throws a ValueError if any of these are importable

    This ensures that when we're loading a neuropod, we're not overwriting any
    existing packages and the neuropod is being loaded correctly.

    :param  package:    The package to verify is not importable.
                        Ex: `petastorm.workers_pool.process_pool`
    """
    while "." in package:
        try:
            importlib.import_module(package)
            raise ValueError(("Package `{}` is importable before loading the neuropod! "
                              "This means that your neuropod package clashes with something already "
                              "accessible by python. Please check your PYTHONPATH and try again").format(package))
        except ImportError:
            pass

        # Remove everything following the last "." (including the ".")
        package = package.rsplit(".", 1)[0]


class PythonNeuropodExecutor(NeuropodExecutor):
    """
    Executes a python neuropod
    """

    def __init__(self, neuropod_path):
        """
        Load a python neuropod

        :param  neuropod_path:  The path to a python neuropod package
        """
        super(PythonNeuropodExecutor, self).__init__(neuropod_path)
        self.neuropod_path = neuropod_path

        # Load the model config
        data_path = os.path.join(neuropod_path, "0", "data")
        with open(os.path.join(neuropod_path, "0", "config.json"), "r") as config_file:
            model_config = json.load(config_file)

        # Load entrypoint info from config
        entrypoint_package_path = model_config["entrypoint_package"]
        entrypoint_fn_name = model_config["entrypoint"]

        # Verify that loading this neuropod won't clash with anything in our existing path
        _validate_not_importable(entrypoint_package_path)

        # Add the neuropod code path to the beginning of the python path
        sys.path.insert(0, os.path.join(neuropod_path, "0", "code"))

        # Import the entrypoint package
        entrypoint_package = importlib.import_module(entrypoint_package_path)

        # Get the entrypoint function and run it with the data path
        self.model = entrypoint_package.__dict__[entrypoint_fn_name](data_path)

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
        out = self.model(**inputs)

        # Make sure everything is a numpy array
        for key, value in out.items():
            if not isinstance(value, np.ndarray):
                raise RuntimeError(
                    "All outputs must be numpy arrays! Output `{}` was of type `{}`".format(
                        key,
                        type(value)))

        return out
