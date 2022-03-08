# Copyright (c) 2020 The Neuropod Authors
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

import atexit
import os
import six
import sys
import tempfile
import uuid
import importlib
import json

import numpy as np

from _neuropod_native_bootstrap.hash_utils import sha256sum
from _neuropod_native_bootstrap.pip_utils import load_deps

# Workaround for https://bugs.python.org/issue32573
if not hasattr(sys, "argv"):
    sys.argv = [""]

# Create the neuropod package symlink directory
SYMLINKS_DIR = tempfile.mkdtemp(suffix=".neuropod_python_symlinks")
open(os.path.join(SYMLINKS_DIR, "__init__.py"), "a").close()

# Add it to our path if necessary
sys.path.insert(0, SYMLINKS_DIR)


def cleanup_symlink():
    # Remove the symlinks (and __init__.py)
    for f in os.listdir(SYMLINKS_DIR):
        os.unlink(os.path.join(SYMLINKS_DIR, f))

    # Delete the directory
    os.rmdir(SYMLINKS_DIR)


# Make sure we clean up this directory at exit
atexit.register(cleanup_symlink)

# Avoid loading the same custom op twice
loaded_op_hashes = set()


class NativePythonExecutor:
    """
    Executes a python neuropod
    """

    def __init__(self, neuropod_path):
        """
        Load a python neuropod

        :param  neuropod_path:  The path to a python neuropod package
        """
        self.neuropod_path = neuropod_path

        # Load the model config
        data_path = os.path.join(neuropod_path, "0", "data")
        with open(os.path.join(neuropod_path, "0", "config.json"), "r") as config_file:
            model_config = json.load(config_file)

        # Load entrypoint info from config
        entrypoint_package_path = model_config["entrypoint_package"]
        entrypoint_fn_name = model_config["entrypoint"]

        # This is running from the native code - set up the requirements if any
        # Note: This is only intended to work when using OPE so this can be problematic
        # when running multiple python models in a single process
        # (which you should try to avoid anyway because of the GIL)
        # For other code paths (e.g. when calling `load_neuropod` from python with `_always_use_native=False`),
        # the user is responsible for ensuring that all dependencies are installed in the environment
        lockfile = os.path.join(neuropod_path, "0", "requirements.lock")
        if os.path.isfile(lockfile):
            load_deps(lockfile)

        # Add the custom op paths to the beginning of the python path
        # Note: there currently isn't a good way to handle multiple custom ops with the same name.
        # Depending on the underlying framework, there can be a global registry that the ops register themselves with
        # If multiple neuropods are loaded that contain ops with the same name, this can cause a conflict or silently
        # use an incorrect op.
        #
        # For complete isolation, out of process execution is the best solution
        custom_op_path = os.path.abspath(os.path.join(neuropod_path, "0", "ops"))

        if os.path.isdir(custom_op_path):
            # Try to avoid silently using the incorrect op
            for item in os.listdir(custom_op_path):
                lib_path = os.path.join(custom_op_path, item)
                lib_hash = sha256sum(lib_path)
                if lib_hash in loaded_op_hashes:
                    # We already loaded this op so it's fine if this is added to the path again
                    # because the op is identical
                    # TODO(vip): This might not be entirely true because of transitive dependencies
                    continue

                loaded_op_hashes.add(lib_hash)

                # If we haven't already loaded the op, make sure there isn't something already loadable with the
                # same name as this op
                op_name = os.path.splitext(item)[0]
                try:
                    importlib.import_module(op_name)
                    raise ValueError(
                        (
                            "Package `{}` is importable before loading the neuropod! "
                            "This means that a custom op in your neuropod package clashes with something already "
                            "accessible by python. Please check your PYTHONPATH and try again"
                        ).format(op_name)
                    )
                except ImportError:
                    pass

            # Add the ops directory to the path
            sys.path.insert(0, custom_op_path)

        # Create a symlink to our code directory
        neuropod_code_path = os.path.abspath(os.path.join(neuropod_path, "0", "code"))
        rand_id = str(uuid.uuid4()).replace("-", "_")
        self.symlink_path = os.path.join(SYMLINKS_DIR, rand_id)
        os.symlink(neuropod_code_path, self.symlink_path)

        # Import the entrypoint package
        if six.PY3:
            # We need to clear the import system caches to make sure it can find the new module
            # See https://docs.python.org/3/library/importlib.html#importlib.import_module
            importlib.invalidate_caches()

        entrypoint_package = importlib.import_module(
            "{}.{}".format(rand_id, entrypoint_package_path)
        )

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
        # Convert bytes to unicode
        for k, v in inputs.items():
            if v.dtype.type == np.bytes_:
                try:
                    inputs[k] = np.char.decode(v, encoding="UTF-8")
                except UnicodeDecodeError:
                    raise ValueError("Error in UTF-8 decoding: {}".format(v))

        out = self.model(**inputs)

        # Make sure everything is a numpy array
        for key, value in out.items():
            if not isinstance(value, np.ndarray):
                raise RuntimeError(
                    "All outputs must be numpy arrays! Output `{}` was of type `{}`".format(
                        key, type(value)
                    )
                )

        # Convert unicode to bytes
        for k, v in out.items():
            if v.dtype.type == np.unicode_:
                out[k] = np.char.encode(v, encoding="UTF-8")

        return out
