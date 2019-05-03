#
# Uber, Inc. (c) 2019
#

import tensorflow as tf

from neuropods.backends.tensorflow.packager import create_tensorflow_neuropod


def create_keras_neuropod(
        neuropod_path,
        model_name,
        sess,
        model,
        input_spec,
        output_spec,
        node_name_mapping=None,
        test_input_data=None,
        test_expected_out=None):
    """
    Packages a Keras model as a neuropod package. Currently, only the TensorFlow backend is supported.

    :param  neuropod_path:      The output neuropod path.

    :param  model_name:         The name of the model.

    :param  sess:               A TensorFlow session containing weights (usually `keras.backend.get_session()`).

    :param  model:              A Keras model object.

    :param  input_spec:         A list of dicts specifying the input to the model. For each input, if shape
                                is set to `None`, no validation is done on the shape. If shape is a tuple, the
                                dimensions of the input are validated against that tuple.  A value of
                                `None` for any of the dimensions means that dimension will not be checked.
                                `dtype` can be any valid numpy datatype string.
                                Ex: [
                                    {"name": "x", "dtype": "float32", "shape": (None,)},
                                    {"name": "y", "dtype": "float32", "shape": (None,)},
                                ]

                                Can be auto-generated using `infer_keras_input_spec()`.

    :param  output_spec:        A list of dicts specifying the output of the model. See the documentation for
                                the `input_spec` parameter for more details.
                                Ex: [
                                    {"name": "out", "dtype": "float32", "shape": (None,)},
                                ]

                                Can be auto-generated using `infer_keras_output_spec()`.

    :param  node_name_mapping:  Optional mapping from a neuropod input/output name to a name of Keras layers
                                Ex: {
                                    "x": "input_1",
                                    "out": "fc1000",
                                }

                                Defaults to using Keras input/output layer names as Neuropod input/output names.

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
    tf_node_mapping = dict()
    if node_name_mapping is not None:
        for name, layer_name in node_name_mapping.items():
            matched_layers = [l for l in model.layers if l.name == layer_name]
            if not matched_layers:
                raise ValueError('No layer named {name} is found in the model.'.format(name=layer_name))
            # layer.output works for input, output and intermediate layers.
            # For input layers it returns the input tensor.
            tf_node_mapping[name] = matched_layers[0].output
    else:
        for name, tensor in zip(model.input_names, model.inputs):
            tf_node_mapping[name] = tensor
        for name, tensor in zip(model.output_names, model.outputs):
            tf_node_mapping[name] = tensor

    graph_def = model.outputs[0].graph.as_graph_def()
    # TODO: validate that output spec only mentions valid nodes
    tf_output_op_names = [tf_node_mapping[t['name']].op.name for t in output_spec]
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=graph_def,
        output_node_names=tf_output_op_names)

    tf_node_name_mapping = {name: tensor.name for name, tensor in tf_node_mapping.items()}

    create_tensorflow_neuropod(
        neuropod_path=neuropod_path,
        model_name=model_name,
        graph_def=frozen_graph_def,
        node_name_mapping=tf_node_name_mapping,
        input_spec=input_spec,
        output_spec=output_spec,
        test_input_data=test_input_data,
        test_expected_out=test_expected_out
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
    reverse_node_name_mapping = {layer_name: name for name, layer_name in (node_name_mapping or dict()).items()}

    input_spec = []
    for name, tensor in zip(model.input_names, model.inputs):
        dims = tuple(d.value for d in tensor.shape.dims[1:])
        # TODO: verify that node_name_mapping provides covers all inputs
        input_spec.append({
            'name': reverse_node_name_mapping[name] if reverse_node_name_mapping else name,
            'dtype': tensor.dtype.name,
            'shape': ('num_inputs',) + dims
        })

    return input_spec


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
    reverse_node_name_mapping = {layer_name: name for name, layer_name in (node_name_mapping or dict()).items()}

    output_spec = []
    for name, tensor in zip(model.output_names, model.outputs):
        dims = tuple(d.value for d in tensor.shape.dims[1:])
        # TODO: verify that node_name_mapping provides covers all outputs
        output_spec.append({
            'name': reverse_node_name_mapping[name] if reverse_node_name_mapping else name,
            'dtype': tensor.dtype.name,
            'shape': ('num_inputs',) + dims
        })

    return output_spec
