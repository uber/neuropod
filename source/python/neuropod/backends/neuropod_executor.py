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

import abc
import six

import numpy as np

from neuropod.utils import config_utils
from neuropod.utils.dtype_utils import get_dtype


def validate_tensors_against_specs(tensors, tensor_specs):
    # All instances of a symbol in a specification must
    # resolve to the same value at runtime. See below for more detail
    symbol_actual_map = {}

    spec_tensor_names = {str(t["name"]) for t in tensor_specs}
    input_tensor_names = set(tensors.keys())

    unknown_tensor_names = input_tensor_names - spec_tensor_names
    if unknown_tensor_names:
        raise ValueError(
            "Tensor name(s) '{}' are not found in the input spec".format(
                ", ".join(unknown_tensor_names)
            )
        )

    # Iterate through all the tensor specs and validate
    # the matching tensor
    for spec in tensor_specs:
        name = spec["name"]
        dtype = get_dtype(spec["dtype"])
        shape = spec["shape"]

        # TODO(yevgeni): treat all tensors as optional. If a particular model requires the input, it will fail therein.
        if name not in tensors:
            # raise ValueError("Missing required tensor: {}".format(name))
            continue

        tensor = tensors[name]

        # Validate the data type
        if tensor.dtype.type != dtype.type:
            raise ValueError(
                "Tensor '{}' is expected to be of type {}, but was of type {}".format(
                    name, dtype, tensor.dtype
                )
            )

        # Validate the number of dimensions
        if len(tensor.shape) != len(shape):
            raise ValueError(
                "Tensor '{}' is expected to have {} dimensions, but had {}".format(
                    name, len(shape), len(tensor.shape)
                )
            )

        # Validate the shape
        for i, (dim, expected) in enumerate(zip(tensor.shape, shape)):
            if expected is None:
                # Any value of dim is okay
                continue
            elif isinstance(expected, six.integer_types):
                # Check that we have the expected number of items
                if dim != expected:
                    raise ValueError(
                        "Dim {} of tensor '{}' is expected to be of size {}, but was of size {}".format(
                            i, name, expected, dim
                        )
                    )
            elif isinstance(expected, six.string_types):
                # `expected` is a symbol
                # Every instance of `expected` should have the same value
                # For example, if a symbol of "num_classes" is used multiple times in the spec,
                # all instances must have the same value
                if expected in symbol_actual_map:
                    # We've seen this symbol before
                    actual_value = symbol_actual_map[expected]

                    # Make sure this usage matches the previous value
                    if dim != actual_value:
                        raise ValueError(
                            (
                                "All dims with expected value '{}' should be the same size. "
                                "Dim {} of tensor '{}' was expected to be of size {}, but was of size {}"
                            ).format(expected, i, name, actual_value, dim)
                        )
                else:
                    # This is the first time we're seeing this symbol
                    # Add it to the map so we can check future occurrences of this symbol
                    symbol_actual_map[expected] = dim

            else:
                raise ValueError(
                    "Invalid value of item in expected shape: {}".format(expected)
                )


@six.add_metaclass(abc.ABCMeta)
class NeuropodExecutor(object):
    """
    Base class for an Executor
    """

    def __init__(self, neuropod_path):
        # Read the neuropod config
        self.neuropod_config = config_utils.read_neuropod_config(neuropod_path)

        # Generate the tensor to device mapping
        self.input_device_mapping = {
            tensor["name"]: self.neuropod_config["input_tensor_device"][tensor["name"]]
            for tensor in self.inputs
        }

    @property
    def name(self):
        """
        Get the name of the loaded neuropod.
        """
        return self.neuropod_config["name"]

    @property
    def platform(self):
        """
        Get the platform of backend of the loaded neuropod.
        """
        return self.neuropod_config["platform"]

    @property
    def inputs(self):
        """
        Get the inputs of the loaded neuropod. Returns a list of dicts representing
        the format of the expected input to the neuropod.

        Ex: [{"name": "x", "dtype": "float32", "shape": [None,]}]
        """
        return self.neuropod_config["input_spec"]

    @property
    def outputs(self):
        """
        Get the outputs of the loaded neuropod. Returns a list of dicts representing
        the format of the output of the neuropod.

        Ex: [{"name": "z", "dtype": "float32", "shape": [None,]}]
        """
        return self.neuropod_config["output_spec"]

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
        # Convert unicode to string
        for k, v in inputs.items():
            if v.dtype.type == np.unicode_:
                inputs[k] = v.astype("str")

        # Validate inputs
        validate_tensors_against_specs(inputs, self.neuropod_config["input_spec"])

        # Run the backend specific inference function
        out = self.forward(inputs)

        # Make sure the key is ascii
        out = {key: value for key, value in out.items()}

        # Convert unicode to string
        for k, v in out.items():
            if v.dtype.type == np.unicode_:
                out[k] = v.astype("str")

        # Validate outputs
        validate_tensors_against_specs(out, self.neuropod_config["output_spec"])

        return out

    @abc.abstractmethod
    def forward(self, inputs):
        """
        Run inference given a set of inputs. See the docstring for `infer`
        """
        raise NotImplementedError("forward must be implemented by subclasses!")

    def __enter__(self):
        # Needed in order to be used as a contextmanager
        return self

    def __exit__(self, *args):
        # Needed in order to be used as a contextmanager
        pass
