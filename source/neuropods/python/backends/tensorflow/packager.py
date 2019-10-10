#
# Uber, Inc. (c) 2018
#

import json
import os
import shutil
import tensorflow as tf

from neuropods.utils.packaging_utils import create_neuropod


def create_tensorflow_neuropod(
        node_name_mapping,
        frozen_graph_path=None,
        graph_def=None,
        init_op_names=[],
        **kwargs):
    """
    Packages a TensorFlow model as a neuropod package.

    :param  neuropod_path:      The output neuropod path

    :param  model_name:         The name of the model

    :param  node_name_mapping:  Mapping from a neuropod input/output name to a node in the graph
                                Ex: {
                                    "x": "some_namespace/in_x:0",
                                    "y": "some_namespace/in_y:0",
                                    "out": "some_namespace/out:0",
                                }

    :param  input_spec:         A list of dicts specifying the input to the model. For each input, if shape
                                is set to `None`, no validation is done on the shape. If shape is a tuple, the
                                dimensions of the input are validated against that tuple.  A value of
                                `None` for any of the dimensions means that dimension will not be checked.
                                `dtype` can be any valid numpy datatype string.
                                Ex: [
                                    {"name": "x", "dtype": "float32", "shape": (None,)},
                                    {"name": "y", "dtype": "float32", "shape": (None,)},
                                ]

    :param  output_spec:        A list of dicts specifying the output of the model. See the documentation for
                                the `input_spec` parameter for more details.
                                Ex: [
                                    {"name": "out", "dtype": "float32", "shape": (None,)},
                                ]

    :param  frozen_graph_path:  The path to a frozen tensorflow graph. If this is not provided, `graph_def` must
                                be set

    :param  graph_def:          A tensorflow GraphDef object. If this is not provided, `frozen_graph_path` must
                                be set

    :param init_op_names:       A list of initialization operator names. These operations are evaluated in the session
                                used for inference right after the session is created. These operators may be used
                                for initialization of variables.


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

    :param  persist_test_data:  Optionally saves the test data within the packaged neuropod. default True.
    """
    # Make sure the inputs are valid
    if (frozen_graph_path is not None and graph_def is not None) or (frozen_graph_path is None and graph_def is None):
        raise ValueError("Exactly one of 'frozen_graph_path' and 'graph_def' must be provided.")

    def packager_fn(neuropod_path):
        # Create a folder to store the model
        neuropod_data_path = os.path.join(neuropod_path, "0", "data")
        os.makedirs(neuropod_data_path)

        if frozen_graph_path is not None:
            # Copy in the frozen graph
            shutil.copyfile(frozen_graph_path, os.path.join(neuropod_data_path, "model.pb"))
        elif graph_def is not None:
            # Write out the frozen graph
            tf.train.write_graph(graph_def, neuropod_data_path, "model.pb", as_text=False)

        # We also need to save the node name mapping so we know how to run the model
        # This is tensorflow specific config so it's not saved in the overall neuropod config
        with open(os.path.join(neuropod_path, "0", "config.json"), "w") as config_file:
            json.dump({
                "node_name_mapping": node_name_mapping,
                "init_op_names": init_op_names if isinstance(init_op_names, list) else [init_op_names],
            }, config_file)

    create_neuropod(
        packager_fn=packager_fn,
        platform="tensorflow",
        **kwargs
    )
