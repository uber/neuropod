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

import tensorflow as tf

from neuropod.backends.tensorflow.packager import create_tensorflow_neuropod
from neuropod.utils.packaging_utils import (
    set_packager_docstring,
    expand_default_kwargs,
)


@set_packager_docstring
@expand_default_kwargs(deps=[create_tensorflow_neuropod])
def create_keras_neuropod(
    sess, model, node_name_mapping=None, input_spec=None, output_spec=None, **kwargs
):
    """
    Packages a Keras model as a neuropod package. Currently, only the TensorFlow backend is supported.

    {common_doc_pre}

    :param  sess:               A TensorFlow session containing weights (usually `keras.backend.get_session()`).

    :param  model:              A Keras model object.

    :param  node_name_mapping:  Optional mapping from a neuropod input/output name to a name of Keras input/output

                                !!! note ""
                                    ***Example***:
                                    ```
                                    {
                                        "x": "input_1",
                                        "out": "fc1000",
                                    }
                                    ```

                                Defaults to using Keras input/output names as neuropod input/output names.

    {common_doc_post}
    """
    if input_spec is None:
        input_spec = infer_keras_input_spec(model, node_name_mapping)
    else:
        _check_spec(input_spec, "input", model.input_names, node_name_mapping)

    if output_spec is None:
        output_spec = infer_keras_output_spec(model, node_name_mapping)
    else:
        _check_spec(output_spec, "output", model.output_names, node_name_mapping)

    tf_node_mapping = dict()
    if node_name_mapping is not None:
        for name, keras_name in node_name_mapping.items():
            if keras_name in model.input_names:
                tf_node_mapping[name] = model.inputs[
                    model.input_names.index(keras_name)
                ]
            elif keras_name in model.output_names:
                tf_node_mapping[name] = model.outputs[
                    model.output_names.index(keras_name)
                ]
            else:
                raise ValueError(
                    "{keras_name} is neither a Keras input name nor a Keras output name."
                    "".format(keras_name=keras_name)
                )
    else:
        for name, tensor in zip(model.input_names, model.inputs):
            tf_node_mapping[name] = tensor
        for name, tensor in zip(model.output_names, model.outputs):
            tf_node_mapping[name] = tensor

    graph_def = model.outputs[0].graph.as_graph_def()
    tf_output_op_names = [
        tf_node_mapping[spec_el["name"]].op.name for spec_el in output_spec
    ]

    # Convert variables to constants is deprecated in Tensorflow 2.2
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/graph_util/convert_variables_to_constants
    convert_variables_to_constants = (
        tf.graph_util.convert_variables_to_constants
        if hasattr(tf.graph_util, "convert_variables_to_constants")
        else tf.compat.v1.graph_util.convert_variables_to_constants
    )

    frozen_graph_def = convert_variables_to_constants(
        sess=sess, input_graph_def=graph_def, output_node_names=tf_output_op_names
    )

    tf_node_name_mapping = {
        name: tensor.name for name, tensor in tf_node_mapping.items()
    }

    create_tensorflow_neuropod(
        graph_def=frozen_graph_def,
        node_name_mapping=tf_node_name_mapping,
        input_spec=input_spec,
        output_spec=output_spec,
        **kwargs
    )


def _check_spec(spec, spec_type, names, node_name_mapping):
    """
    Function checking whether specification only references allowed set of names.
    """
    for spec_el in spec:
        name = spec_el["name"]
        if node_name_mapping:
            keras_name = node_name_mapping.get(name)
            if keras_name is None:
                raise ValueError(
                    "{spec_type} {name} is not mapped in node_name_mapping."
                    "".format(spec_type=spec_type, name=name).capitalize()
                )
            if keras_name not in names:
                raise ValueError(
                    "{spec_type} {name} mapped to {keras_name} is not in model {spec_type}s."
                    "".format(
                        spec_type=spec_type, name=name, keras_name=keras_name
                    ).capitalize()
                )
        else:
            if name not in names:
                raise ValueError(
                    "{spec_type} {name} is not in model {spec_type}s."
                    "".format(spec_type=spec_type, name=name).capitalize()
                )


def infer_keras_input_spec(model, node_name_mapping=None):
    """
    Infers input schema of a Keras model.

    :param  model:              A Keras model object.

    :param  node_name_mapping:  Optional mapping from a neuropod input/output name to a name of Keras layers
                                Ex: {
                                    "x": "input_1",
                                    "out": "fc1000",
                                }

                                Defaults to using Keras input/output layer names as Neuropod input/output names.

    :returns:                   An input spec suitable to be passed to `create_tensorflow_keras_neuropod()`.
    """
    return _infer_keras_spec(model.input_names, model.inputs, node_name_mapping)


def infer_keras_output_spec(model, node_name_mapping=None):
    """
    Infers output schema of a Keras model.

    :param  model:              A Keras model object.

    :param  node_name_mapping:  Optional mapping from a neuropod input/output name to a name of Keras layers
                                Ex: {
                                    "x": "input_1",
                                    "out": "fc1000",
                                }

                                Defaults to using Keras input/output layer names as Neuropod input/output names.

    :returns:                   An output spec suitable to be passed to `create_tensorflow_keras_neuropod()`.
    """
    return _infer_keras_spec(model.output_names, model.outputs, node_name_mapping)


def _infer_keras_spec(names, tensors, node_name_mapping):
    """
    Function implementing the spec inference for either input or output.
    """
    reverse_node_name_mapping = {
        keras_name: name for name, keras_name in (node_name_mapping or dict()).items()
    }

    spec = []
    for keras_name, tensor in zip(names, tensors):
        # Skip the first dimension - batch size.
        dims = tuple(d.value for d in tensor.shape.dims[1:])

        if reverse_node_name_mapping:
            # If the node_name_mapping is defined, it must cover all inputs and outputs.
            name = reverse_node_name_mapping.get(keras_name)
            if name is None:
                raise ValueError(
                    "Keras input/output layer {name} is not covered by node_name_mapping."
                    "".format(name=keras_name)
                )
        else:
            name = keras_name

        spec.append(
            {"name": name, "dtype": tensor.dtype.name, "shape": ("batch_size",) + dims}
        )

    return spec
