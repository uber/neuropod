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
import numpy as np
import os
import six
import tensorflow as tf

from neuropod.backends.neuropod_executor import NeuropodExecutor
from neuropod.utils.dtype_utils import get_dtype
from neuropod.utils.hash_utils import sha256sum

# Avoid loading the same custom op twice
loaded_op_hashes = set()


class TensorflowFrozenGraphNeuropodExecutor(NeuropodExecutor):
    """
    Executes a Tensorflow neuropod that contains a frozen graph
    """

    def __init__(self, neuropod_path, load_custom_ops=True):
        """
        Load a Tensorflow neuropod

        :param  neuropod_path:  The path to a python neuropod package
        """
        super(TensorflowFrozenGraphNeuropodExecutor, self).__init__(neuropod_path)

        # Load custom ops (if any)
        if load_custom_ops and "custom_ops" in self.neuropod_config:
            for op in self.neuropod_config["custom_ops"]:
                lib_path = os.path.join(neuropod_path, "0", "ops", op)
                lib_hash = sha256sum(lib_path)
                if lib_hash not in loaded_op_hashes:
                    tf.load_op_library(str(lib_path))
                    loaded_op_hashes.add(lib_hash)

        gfile = tf.gfile if hasattr(tf, "gfile") else tf.io.gfile
        # Load the model
        with gfile.GFile(
            os.path.join(neuropod_path, "0", "data", "model.pb"), "rb"
        ) as f:
            graph_def = (
                tf.GraphDef() if hasattr(tf, "GraphDef") else tf.compat.v1.GraphDef()
            )
            graph_def.ParseFromString(f.read())

        # Load the TF specific config
        with open(os.path.join(neuropod_path, "0", "config.json"), "r") as config_file:
            model_config = json.load(config_file)

            # Get the node name mapping and store it
            self.node_name_mapping = model_config["node_name_mapping"]

            # Make sure every node in the mapping ends with `:index`
            for k, v in self.node_name_mapping.items():
                if ":" not in v:
                    self.node_name_mapping[k] = v + ":0"

            init_op_names = model_config["init_op_names"]

        # Setup the graph from the definition
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")
            init_ops = [
                self.graph.get_operation_by_name(op_name) for op_name in init_op_names
            ]

        # Create a session
        session = tf.Session if hasattr(tf, "Session") else tf.compat.v1.Session
        self.sess = session(graph=self.graph)
        self.sess.run(init_ops)

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

        # get the input and output nodes
        output_dict = {}
        feed_dict = {}

        # Get the output nodes
        for node in self.neuropod_config["output_spec"]:
            neuropod_name = node["name"]

            # Get the graph node
            tf_name = self.node_name_mapping[neuropod_name]
            tf_node = self.graph.get_tensor_by_name(tf_name)

            # Add it to the output nodes
            output_dict[neuropod_name] = tf_node

        # Get the input nodes
        for node in self.neuropod_config["input_spec"]:
            neuropod_name = node["name"]

            # TODO(yevgeni): treat all input fields as optional at the neuropod level. If a model
            # requires a missing field it will fail therein.
            if neuropod_name not in inputs:
                continue

            # Get the graph node
            tf_name = self.node_name_mapping[neuropod_name]
            tf_node = self.graph.get_tensor_by_name(tf_name)

            # Add it to the feed_dict
            feed_dict[tf_node] = inputs[neuropod_name]

        # Run inference
        outputs = self.sess.run(output_dict, feed_dict=feed_dict)

        # TensorFlow returns string tensors with type object
        for spec in self.neuropod_config["output_spec"]:
            name = spec["name"]
            dtype = get_dtype(spec["dtype"])
            if (
                dtype.type == np.str_
                and outputs[name].dtype == "object"
                and type(outputs[name].item(0)) == six.binary_type
            ):
                # If the tensor is supposed to be of type string, is of type object, and contains strings
                outputs[name] = outputs[name].astype("str")

        return outputs


class TensorflowSavedModelNeuropodExecutor(NeuropodExecutor):
    """
    Executes a Tensorflow neuropod that contains a SavedModel

    Note: only tested with TF2
    """

    def __init__(self, neuropod_path, load_custom_ops=True):
        """
        Load a Tensorflow neuropod

        :param  neuropod_path:  The path to a python neuropod package
        """
        super(TensorflowSavedModelNeuropodExecutor, self).__init__(neuropod_path)

        # Load custom ops (if any)
        if load_custom_ops and "custom_ops" in self.neuropod_config:
            for op in self.neuropod_config["custom_ops"]:
                lib_path = os.path.join(neuropod_path, "0", "ops", op)
                lib_hash = sha256sum(lib_path)
                if lib_hash not in loaded_op_hashes:
                    tf.load_op_library(str(lib_path))
                    loaded_op_hashes.add(lib_hash)

        # Load the savedmodel
        loaded = tf.saved_model.load(
            os.path.join(neuropod_path, "0", "data", "savedmodel")
        )
        self._model = loaded.signatures["serving_default"]

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
        # Convert to tensors
        transformed = {k: tf.convert_to_tensor(v) for k, v in inputs.items()}

        # Run inference
        out = self._model(**transformed)

        # Convert to numpy arrays and return
        return {k: v.numpy() for k, v in out.items()}


class TensorflowNeuropodExecutor(NeuropodExecutor):
    """
    Executes a Tensorflow neuropod
    """

    def __init__(self, neuropod_path, **kwargs):
        """
        Load a Tensorflow neuropod

        :param  neuropod_path:  The path to a python neuropod package
        """
        super(TensorflowNeuropodExecutor, self).__init__(neuropod_path)

        # Load the model
        if os.path.exists(os.path.join(neuropod_path, "0", "data", "model.pb")):
            # Frozen graph
            self._model = TensorflowFrozenGraphNeuropodExecutor(neuropod_path, **kwargs)
        else:
            # Saved model
            self._model = TensorflowSavedModelNeuropodExecutor(neuropod_path, **kwargs)

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
        # Run inference
        return self._model.forward(inputs)
