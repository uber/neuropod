#
# Uber, Inc. (c) 2018
#

from neuropods.backends import config_utils


def load_neuropod(neuropod_path, **kwargs):
    """
    Load a neuropod package. Returns a NeuropodExecutor

    :param  neuropod_path   The path to a neuropod package
    """
    # Figure out what type of neuropod this is
    neuropod_config = config_utils.read_neuropod_config(neuropod_path)
    platform = neuropod_config["platform"]

    if platform == "python":
        from neuropods.backends.python.executor import PythonNeuropodExecutor
        return PythonNeuropodExecutor(neuropod_path, **kwargs)
    elif platform == "torchscript":
        from neuropods.backends.torchscript.executor import TorchScriptNeuropodExecutor
        return TorchScriptNeuropodExecutor(neuropod_path, **kwargs)
    elif platform == "tensorflow":
        from neuropods.backends.tensorflow.executor import TensorflowNeuropodExecutor
        return TensorflowNeuropodExecutor(neuropod_path, **kwargs)
    else:
        raise ValueError("Invalid platform found in neuropod config: {}".format(platform))


if __name__ == '__main__':
    import argparse
    from six.moves import cPickle as pickle

    parser = argparse.ArgumentParser(description='Load a model and run inference with provided sample data.')
    parser.add_argument('--neuropod-path', help='The path to a neuropod to load', required=True)
    parser.add_argument('--input-pkl-path', help='The path to sample input data for the model', required=True)
    parser.add_argument('--output-pkl-path', help='Where to write the output of the model', default=None)
    parser.add_argument('--use-native', help=('Use the native bindings to run the model. This option can currently '
                                              'only be used when running from the `source/python` directory within '
                                              'the Neuropods source tree.'), default=False, action='store_true')
    args = parser.parse_args()

    def run_model(model):
        # Load the input data
        with open(args.input_pkl_path, 'rb') as pkl:
            input_data = pickle.load(pkl)

        out = model.infer(input_data)

        # Save the output data to the requested path
        if args.output_pkl_path is not None:
            with open(args.output_pkl_path, 'wb') as pkl:
                pickle.dump(out, pkl)

    if args.use_native:
        import os
        from neuropods_native import Neuropod as NeuropodNative
        # We need to override the default backend lookup paths to point to the shared objects in
        # the bazel bin directory
        # TODO(vip): Do this in a place that only affects the tests
        DEFAULT_BACKEND_OVERRIDES = {
            "tensorflow": "../bazel-bin/neuropods/backends/tensorflow/libneuropod_tensorflow_backend.so",
            "python": "../bazel-bin/neuropods/backends/python_bridge/libneuropod_pythonbridge_backend.so",
            "pytorch": "../bazel-bin/neuropods/backends/python_bridge/libneuropod_pythonbridge_backend.so",
            "torchscript": "../bazel-bin/neuropods/backends/torchscript/libneuropod_torchscript_backend.so",
        }

        if not os.path.isdir("../bazel-bin"):
            raise ValueError("The `--use-native` option can currently only be used when running from the `source/python` directory within the Neuropods source tree. "
                             "Please also ensure that the native libraries have been built before using this flag.")
        model = NeuropodNative(args.neuropod_path, DEFAULT_BACKEND_OVERRIDES)
        run_model(model)
    else:
        with load_neuropod(args.neuropod_path) as model:
            run_model(model)
