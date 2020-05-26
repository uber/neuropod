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
    node_name_mapping,
    frozen_graph_path=None,
    graph_def=None,
    init_op_names=[],
    **kwargs
):
    """
    Packages a TensorFlow model as a neuropod package.

    {common_doc_pre}

    :param  node_name_mapping:  Mapping from a neuropod input/output name to a node in the graph. The `:0` is
                                optional.

                                !!! note ""
                                    ***Example***:
                                    ```
                                    {
                                        "x": "some_namespace/in_x:0",
                                        "y": "some_namespace/in_y:0",
                                        "out": "some_namespace/out:0",
                                    }
                                    ```

    :param  frozen_graph_path:  The path to a frozen tensorflow graph. If this is not provided, `graph_def` must
                                be set

    :param  graph_def:          A tensorflow `GraphDef` object. If this is not provided, `frozen_graph_path` must
                                be set

    :param init_op_names:       A list of initialization operator names. These operations are evaluated in the session
                                used for inference right after the session is created. These operators may be used
                                for initialization of variables.

    {common_doc_post}
    """
    # Make sure the inputs are valid
    if (frozen_graph_path is not None and graph_def is not None) or (
        frozen_graph_path is None and graph_def is None
    ):
        raise ValueError(
            "Exactly one of 'frozen_graph_path' and 'graph_def' must be provided."
        )

    # Create a folder to store the model
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    os.makedirs(neuropod_data_path)

    if frozen_graph_path is not None:
        # Copy in the frozen graph
        shutil.copyfile(frozen_graph_path, os.path.join(neuropod_data_path, "model.pb"))
    elif graph_def is not None:
        # Write out the frozen graph. tf.io.write_graph is an alias to tf.train.write_graph but is safer
        # as it is also present in tensorflow 2.
        tf.io.write_graph(graph_def, neuropod_data_path, "model.pb", as_text=False)

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
