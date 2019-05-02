#
# Uber, Inc. (c) 2019
#

from neuropods.backends.tensorflow.packager import create_tensorflow_neuropod


def convert_keras_model_to_tensorflow_neuropod(
        neuropod_path,
        model_name,
        model,
        test_input_data=None,
        test_expected_out=None):
    """
    Packages a TensorFlow model as a neuropod package.

    :param  neuropod_path:      The output neuropod path

    :param  model_name:         The name of the model

    :param  model:              A Keras model object.

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
    node_name_mapping = dict()
    for name, tensor in zip(model.input_names, model.inputs):
        node_name_mapping[name] = tensor.name
    for name, tensor in zip(model.output_names, model.outputs):
        node_name_mapping[name] = tensor.name

    input_spec = []
    for name, tensor in zip(model.input_names, model.inputs):
        dims = tuple(d.value for d in tensor.shape.dims[1:])
        input_spec.append({
            'name': name,
            'dtype': tensor.dtype.name,
            'shape': ('num_inputs',) + dims
        })

    output_spec = []
    for name, tensor in zip(model.output_names, model.outputs):
        dims = tuple(d.value for d in tensor.shape.dims[1:])
        output_spec.append({
            'name': name,
            'dtype': tensor.dtype.name,
            'shape': ('num_inputs',) + dims
        })

    graph_def = model._graph.as_graph_def()

    create_tensorflow_neuropod(
        neuropod_path=neuropod_path,
        model_name=model_name,
        graph_def=graph_def,
        node_name_mapping=node_name_mapping,
        input_spec=input_spec,
        output_spec=output_spec,
        test_input_data=test_input_data,
        test_expected_out=test_expected_out
    )
