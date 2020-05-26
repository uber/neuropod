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

import numpy as np
import tensorflow as tf
from collections import defaultdict
from six import string_types

from neuropod.packagers import create_tensorflow_neuropod
from neuropod.utils.dtype_utils import get_dtype


def _placeholdes_from_input_spec(input_spec, input_prefix="INPUT_API"):
    """Creates tf.placeholder for each of the entries in the input_spec. Returns a dictionary with the
    mapping neuropod input name to the fully qualified tensorflow tensor name"""
    node_name_mapping = dict()
    with tf.name_scope(input_prefix):
        for tensor_spec in input_spec:
            name = tensor_spec["name"]

            symbolic_shape = tensor_spec["shape"]
            # When a shape is defined by a string symbol, it means it's a variable intput.
            shape = tuple(
                (None if isinstance(d, string_types) else d for d in symbolic_shape)
            )

            # Translate string name to numpy type and the to TF dtype
            numpy_dtype = tensor_spec["dtype"]
            tf_dtype = tf.as_dtype(get_dtype(numpy_dtype))

            placeholder = tf.placeholder(tf_dtype, name=name, shape=shape)

            node_name_mapping[name] = placeholder.name

    return node_name_mapping


def _random_from_output_spec(output_spec, output_prefix="OUTPUT_API"):
    """Adds random matrix generators based on the output spec. Symbolic dimensions in shape definition are respected."""
    node_name_mapping = dict()

    # Arbitrary choice of the number of elements in a variable size dimension: 1 to 100
    def toss_random_dim():
        return np.random.randint(1, 100)

    symbol_value = defaultdict(toss_random_dim)

    with tf.name_scope(output_prefix):

        for tensor_spec in output_spec:
            name = tensor_spec["name"]

            symbolic_shape = tensor_spec["shape"]

            # Randomize variable sized dimensions.
            resolved_shape = tuple()
            for d in symbolic_shape:
                if isinstance(d, string_types):
                    resolved_shape += (symbol_value[d],)
                elif d is None:
                    resolved_shape += (toss_random_dim(),)
                else:
                    resolved_shape += (d,)

            numpy_dtype = tensor_spec["dtype"]
            tf_dtype = tf.as_dtype(get_dtype(numpy_dtype))

            if numpy_dtype != "string":
                # Integers need `maxval=` to be specified explicitly. Also, random_uniform does not support all
                # integer types.
                if tf_dtype.is_integer:
                    output_tensor = tf.cast(
                        tf.random_uniform(
                            shape=resolved_shape,
                            maxval=tf_dtype.max,
                            dtype=tf.int64,
                            name=name,
                        )
                        % tf_dtype.max,
                        tf_dtype,
                    )
                else:
                    output_tensor = tf.random_uniform(
                        shape=resolved_shape, dtype=tf_dtype, name=name
                    )
            else:
                # We just convert random floats to strings
                output_tensor = tf.as_string(
                    tf.random_uniform(shape=resolved_shape, dtype=tf.float32, name=name)
                )

            node_name_mapping[name] = output_tensor.name

    return node_name_mapping


def randomify_neuropod(output_path, input_spec, output_spec):
    """Uses neuropod input and output specs to automatically generate a neuropod package that complies to the spec and
    produces random outputs.

    This neuropod can be used as a stub, for testing purposes. A Tensorflow engine is used in the neuropod generated.

    :param output_path: Output path where the neuropod will be written to.
    :param input_spec: Neuropod input spec structure (a list of dictionaries)
    :param output_spec: Neuropod output spec structure (a list of dictionaries)
    :return:
    """
    g = tf.Graph()
    with g.as_default():
        # Create a placeholder of a corresponding shape for each of the inputs in the input spec
        node_name_mapping = _placeholdes_from_input_spec(input_spec)

        # For each output, we create a random matrix generator of an appropriate type and size.
        node_name_mapping.update(_random_from_output_spec(output_spec))

    graph_def = g.as_graph_def()

    create_tensorflow_neuropod(
        neuropod_path=output_path,
        model_name="random_model",
        graph_def=graph_def,
        node_name_mapping=node_name_mapping,
        input_spec=input_spec,
        output_spec=output_spec,
    )

    return output_path
