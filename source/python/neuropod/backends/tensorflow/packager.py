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

import json
import os
import shutil
import tensorflow as tf

from neuropod.utils.packaging_utils import packager


@packager(platform="tensorflow")
def create_tensorflow_neuropod(
    neuropod_path,
    input_spec,
    output_spec,
    node_name_mapping=None,
    frozen_graph_path=None,
    graph_def=None,
    saved_model_dir=None,
    trackable_obj=None,
    init_op_names=[],
    **kwargs
):
    """
    Packages a TensorFlow model as a neuropod package.

    {common_doc_pre}

    :param  node_name_mapping:  Mapping from a neuropod input/output name to a node in the graph. The `:0` is
                                optional. Required unless using a saved model.

                                !!! note ""
                                    ***Example***:
                                    ```
                                    {
                                        "x": "some_namespace/in_x:0",
                                        "y": "some_namespace/in_y:0",
                                        "out": "some_namespace/out:0",
                                    }
                                    ```

    :param  frozen_graph_path:  The path to a frozen tensorflow graph. Exactly one of `frozen_graph_path`, `graph_def`, `saved_model_dir`
                                and `trackable_obj` must be provided.

    :param  graph_def:          A tensorflow `GraphDef` object. Exactly one of `frozen_graph_path`, `graph_def`, `saved_model_dir`
                                and `trackable_obj` must be provided.

    :param  saved_model_dir:    The path to a tensorflow saved model dir. Exactly one of `frozen_graph_path`, `graph_def`, `saved_model_dir`
                                and `trackable_obj` must be provided.
                                Note: this is only tested with TF 2.x at the moment

    :param  trackable_obj:      A trackable object that can be passed to `tf.saved_model.save`. For more control over the
                                saved model, you can create one yourself and pass in the path using `saved_model_dir`.
                                Exactly one of `frozen_graph_path`, `graph_def`, `saved_model_dir` and `trackable_obj` must be provided.
                                Note: this is only tested with TF 2.x at the moment

    :param init_op_names:       A list of initialization operator names. These operations are evaluated in the session
                                used for inference right after the session is created. These operators may be used
                                for initialization of variables.

    {common_doc_post}
    """
    # Make sure the inputs are valid
    # fmt: off
    if sum([frozen_graph_path is not None, graph_def is not None, saved_model_dir is not None, trackable_obj is not None]) != 1:
        raise ValueError(
            "Exactly one of `frozen_graph_path`, `graph_def`, `saved_model_dir` and `trackable_obj` must be provided."
        )
    # fmt: on

    # Create a folder to store the model
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    os.makedirs(neuropod_data_path)

    # Copy/export the model into the neuropod package
    if frozen_graph_path is not None:
        # Copy in the frozen graph
        shutil.copyfile(frozen_graph_path, os.path.join(neuropod_data_path, "model.pb"))
    elif graph_def is not None:
        # Write out the frozen graph. tf.io.write_graph is an alias to tf.train.write_graph but is safer
        # as it is also present in tensorflow 2.
        tf.io.write_graph(graph_def, neuropod_data_path, "model.pb", as_text=False)
    elif saved_model_dir is not None:
        # Copy the saved model in if we have it
        shutil.copytree(saved_model_dir, os.path.join(neuropod_data_path, "savedmodel"))
    elif trackable_obj is not None:
        # Save trackable_obj to a SavedModel
        saved_model_dir = os.path.join(neuropod_data_path, "savedmodel")
        tf.saved_model.save(trackable_obj, saved_model_dir)

    # Validate the args
    if saved_model_dir is not None:
        # Load the model and make sure its inputs and outputs match the provided spec
        loaded = tf.saved_model.load(saved_model_dir)

        # Get the input SignatureDef from the SavedModel
        structured_inputs = loaded.signatures[
            "serving_default"
        ].structured_input_signature[1]

        # Make sure the inputs match the spec
        actual_inputs = set(name for name in structured_inputs)
        expected_inputs = set(tensor["name"] for tensor in input_spec)
        missing_inputs = expected_inputs - actual_inputs
        extra_inputs = actual_inputs - expected_inputs

        if len(missing_inputs) > 0:
            raise ValueError(
                "The supplied SavedModel is missing the following inputs in the spec: `{}`".format(
                    missing_inputs
                )
            )

        if len(extra_inputs) > 0:
            raise ValueError(
                "The supplied SavedModel expects inputs that are not in the spec: `{}`".format(
                    extra_inputs
                )
            )

        # Make sure the model outputs are a superset of the spec
        actual_outputs = set(
            name for name in loaded.signatures["serving_default"].structured_outputs
        )
        expected_outputs = set(tensor["name"] for tensor in output_spec)
        missing_outputs = expected_outputs - actual_outputs

        if len(missing_outputs) > 0:
            raise ValueError(
                "The supplied SavedModel does not return the following outputs in the spec: `{}`".format(
                    missing_outputs
                )
            )

    else:
        # We require a node name mapping when using a frozen graph or a graphdef
        if node_name_mapping is None:
            raise ValueError(
                "node_name_mapping is required when using `frozen_graph_path` or `graph_def"
            )

        # Make sure we have mappings for everything in the spec
        expected_keys = set()
        for spec in [input_spec, output_spec]:
            for tensor in spec:
                expected_keys.add(tensor["name"])

        actual_keys = set(node_name_mapping.keys())
        missing_keys = expected_keys - actual_keys

        if len(missing_keys) > 0:
            raise ValueError(
                "Expected an item in `node_name_mapping` for every tensor in input_spec and output_spec. Missing: `{}`".format(
                    missing_keys
                )
            )

    # We also need to save the node name mapping so we know how to run the model
    # This is tensorflow specific config so it's not saved in the overall neuropod config
    with open(os.path.join(neuropod_path, "0", "config.json"), "w") as config_file:
        json.dump(
            {
                "node_name_mapping": node_name_mapping,
                "init_op_names": init_op_names
                if isinstance(init_op_names, list)
                else [init_op_names],
            },
            config_file,
        )
