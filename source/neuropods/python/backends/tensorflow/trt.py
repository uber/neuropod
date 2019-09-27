#
# Uber, Inc. (c) 2019
#

import logging
import time
import tensorflow as tf

logger = logging.getLogger(__name__)

# Try importing the TensorRT converter
try:
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
except ImportError:
    trt = None

def is_trt_available():
    return trt is not None

# Inspired by https://github.com/tensorflow/tensorrt/blob/master/tftrt/examples/object_detection/object_detection.py#L328
def trt_optimize(
    graph_def=None,
    frozen_graph_path=None,
    nodes_blacklist=None,
    precision_mode="FP32",
    minimum_segment_size=2,
    max_workspace_size_bytes=1 << 32,
    is_dynamic_op=True,
    maximum_cached_engines=100,
    **kwargs):
    """
    Optimizes a TF model using TensorRT and returns a GraphDef

    :param  graph_def:                  A tensorflow GraphDef object. If this is not provided, `frozen_graph_path` must be set
    :param  frozen_graph_path:          The path to a frozen tensorflow graph. If this is not provided, `graph_def` must be set
    :param  nodes_blacklist:            A required list of nodes for TensorRT to ignore. These usually include the outputs of your model.
    :param  precision_mode:             The precision of the final graph. Quanization is not yet supported so the only allowed value is `FP32`
    :param  minimum_segment_size:       An integer representing the minimum segment size to use for TensorRT graph segmentation
    :param  max_workspace_size_bytes:   An integer representing the max workspace size for TensorRT optimization
    :param  is_dynamic_op:              Whether or not to allow dynamic shapes
    :param  maximum_cached_engines:     An integer represenging the number of TRT engines that can be stored in the cache.

    See https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#tf-trt-api for more information on the TensorRT params
    """

    if not is_trt_available():
        raise ValueError("Could not import the TensorRT converter. Try upgrading to a newer version of TensorFlow")

    # Validate inputs
    if (frozen_graph_path is not None and graph_def is not None) or (frozen_graph_path is None and graph_def is None):
        raise ValueError("Exactly one of 'frozen_graph_path' and 'graph_def' must be provided.")

    if nodes_blacklist is None:
        raise ValueError("`nodes_blacklist` must be provided")

    # Load a frozen graph if necessary
    if frozen_graph_path is not None and graph_def is None:
        # Load the model
        with tf.gfile.GFile(frozen_graph_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

    if precision_mode != "FP32":
        raise ValueError("TensorRT quantization is not yet supported. The only supported precision_mode is `FP32`")

    # Store the original size and num nodes
    graph_size = len(graph_def.SerializeToString())
    num_nodes = len(graph_def.node)

    logger.info("Starting TRT optimization...")
    start_time = time.time()
    converter = trt.TrtGraphConverter(
        input_graph_def=graph_def,
        nodes_blacklist=nodes_blacklist,
        max_workspace_size_bytes=max_workspace_size_bytes,
        precision_mode=precision_mode,
        minimum_segment_size=minimum_segment_size,
        is_dynamic_op=is_dynamic_op,
        maximum_cached_engines=maximum_cached_engines,
        **kwargs)
    graph_def = converter.convert()
    end_time = time.time()

    # Log some optimized info
    logger.info("TRT optimization complete. Took {} seconds".format(end_time - start_time))
    logger.info("graph_size(MB)(native_tf): {}".format(float(graph_size)/(1<<20)))
    logger.info("graph_size(MB)(trt): {}".format(float(len(graph_def.SerializeToString()))/(1<<20)))
    logger.info("num_nodes(native_tf): {}".format(num_nodes))
    logger.info("num_nodes(tftrt_total): {}".format(len(graph_def.node)))
    logger.info("num_nodes(trt_only): {}".format(len([1 for n in graph_def.node if str(n.op)=='TRTEngineOp'])))

    return graph_def
