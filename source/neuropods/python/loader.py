#
# Uber, Inc. (c) 2018
#

from neuropods.backends import config_utils


def load_neuropod(neuropod_path, **kwargs):
    """
    Load a neuropod package. Returns a NeuropodExecutor

    :param  neuropod_path       The path to a neuropod package
    :param  visible_gpu:        The index of the GPU that this Neuropod should run on (if any).
                                This is either `None` or a nonnegative integer. Setting this
                                to `None` will attempt to run this model on CPU.
    :param  load_custom_ops:    Whether or not to load custom ops included in the model.
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
    parser.add_argument('--args-pkl-path', help='The path to kwargs for `load_neuropod`.', default={})
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

    with open(args.args_pkl_path, 'rb') as pkl:
        load_neuropod_kwargs = pickle.load(pkl)

    if args.use_native:
        import os
        from neuropods_native import Neuropod as NeuropodNative

        model = NeuropodNative(args.neuropod_path, **load_neuropod_kwargs)
        run_model(model)
    else:
        with load_neuropod(args.neuropod_path, **load_neuropod_kwargs) as model:
            run_model(model)
