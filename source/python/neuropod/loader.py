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
import numpy as np

from neuropod.registry import _REGISTERED_BACKENDS

# Add the script's directory to the PATH so we can find the worker binary
os.environ["PATH"] += ":" + os.path.dirname(os.path.realpath(__file__))


def _convert_native_shape_to_list(dims):
    """
    Takes a list of `neuropod_native.Dimension` objects and converts to a list of python types
    """
    out = []
    for dim in dims:
        if dim.value == -2:
            # It's a symbol
            out.append(dim.symbol)
        elif dim.value == -1:
            # Any shape is okay
            out.append(None)
        else:
            out.append(dim.value)

    return out


class NativeNeuropodExecutor:
    """
    Executes a Neuropod using the native bindings
    """

    def __init__(self, neuropod_path, **kwargs):
        """
        Load a Neuropod using the native bindings

        :param  neuropod_path:  The path to a neuropod package
        """
        # Load the model
        from neuropod.neuropod_native import Neuropod as NeuropodNative

        self.model = NeuropodNative(
            neuropod_path, _REGISTERED_BACKENDS, use_ope=True, **kwargs
        )

    @property
    def name(self):
        """
        Get the name of the loaded neuropod.
        """
        return self.model.get_name()

    @property
    def platform(self):
        """
        Get the platform of backend of the loaded neuropod.
        """
        return self.model.get_platform()

    @property
    def inputs(self):
        """
        Get the inputs of the loaded neuropod. Returns a list of dicts representing
        the format of the expected input to the neuropod.

        Ex: [{"name": "x", "dtype": "float32", "shape": [None,]}]
        """
        out = []
        for item in self.model.get_inputs():
            out.append(
                {
                    "name": item.name,
                    "dtype": item.type.name,
                    "shape": _convert_native_shape_to_list(item.dims),
                }
            )

        return out

    @property
    def outputs(self):
        """
        Get the outputs of the loaded neuropod. Returns a list of dicts representing
        the format of the output of the neuropod.

        Ex: [{"name": "z", "dtype": "float32", "shape": [None,]}]
        """
        out = []
        for item in self.model.get_outputs():
            out.append(
                {
                    "name": item.name,
                    "dtype": item.type.name,
                    "shape": _convert_native_shape_to_list(item.dims),
                }
            )

        return out

    def infer(self, inputs):
        """
        Run inference using the specifed inputs.

        :param  inputs:     A dict mapping input names to values. This must match the input
                            spec in the neuropod config for the loaded model.
                            Ex: {'x1': np.array([5]), 'x2': np.array([6])}
                            *Note:* all the keys in this dict must be strings and all the
                            values must be numpy arrays

        :returns:   A dict mapping output names to values. This is checked to ensure that it
                    matches the spec in the neuropod config for the loaded model. All the keys
                    in this dict are strings and all the values are numpy arrays.
        """
        # Convert unicode to bytes before running inference
        for key, value in inputs.items():
            if value.dtype.type == np.unicode_:
                inputs[key] = np.char.encode(value, encoding="UTF-8")

        out = self.model.infer(inputs)

        # Convert bytes to unicode
        for key, value in out.items():
            if value.dtype.type == np.bytes_:
                out[key] = np.char.decode(value, encoding="UTF-8")

        return out

    def __enter__(self):
        # Needed in order to be used as a contextmanager
        return self

    def __exit__(self, *args):
        # Needed in order to be used as a contextmanager
        pass


def load_neuropod(neuropod_path, _always_use_native=True, **kwargs):
    """
    Load a neuropod package. Returns a NeuropodExecutor

    :param  neuropod_path       The path to a neuropod package
    :param  visible_gpu:        The index of the GPU that this Neuropod should run on (if any).
                                This is either `None` or a nonnegative integer. Setting this
                                to `None` will attempt to run this model on CPU.
    :param  load_custom_ops:    Whether or not to load custom ops included in the model.
    """
    if _always_use_native:
        return NativeNeuropodExecutor(neuropod_path, **kwargs)
    else:
        raise ValueError(
            "_always_use_native=False has been removed (after having a deprecation warning for 3 months). "
            "It was originally intended to be a workaround for edge cases as the native implementation "
            "was being built. Please remove `_always_use_native=False` as an argument to `load_neuropod`. "
            "This means that Neuropod will use the native code path to run inference (the same code path "
            "used by Neuropod from C++. Java, C, Go, etc.)"
        )


if __name__ == "__main__":
    import argparse
    from six.moves import cPickle as pickle

    parser = argparse.ArgumentParser(
        description="Load a model and run inference with provided sample data."
    )
    parser.add_argument(
        "--neuropod-path", help="The path to a neuropod to load", required=True
    )
    parser.add_argument(
        "--input-pkl-path",
        help="The path to sample input data for the model",
        required=True,
    )
    parser.add_argument(
        "--args-pkl-path", help="The path to kwargs for `load_neuropod`.", default={}
    )
    parser.add_argument(
        "--output-pkl-path", help="Where to write the output of the model", default=None
    )
    args = parser.parse_args()

    def run_model(model):
        # Load the input data
        with open(args.input_pkl_path, "rb") as pkl:
            input_data = pickle.load(pkl)

        out = model.infer(input_data)

        # Save the output data to the requested path
        if args.output_pkl_path is not None:
            with open(args.output_pkl_path, "wb") as pkl:
                pickle.dump(out, pkl)

    with open(args.args_pkl_path, "rb") as pkl:
        load_neuropod_kwargs = pickle.load(pkl)

    with load_neuropod(args.neuropod_path, **load_neuropod_kwargs) as model:
        run_model(model)
