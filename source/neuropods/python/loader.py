#
# Uber, Inc. (c) 2018
#

from neuropods.backends import config_utils


def load_neuropod(neuropod_path):
    """
    Load a neuropod package. Returns a NeuropodExecutor

    :param  neuropod_path   The path to a neuropod package
    """
    # Figure out what type of neuropod this is
    neuropod_config = config_utils.read_neuropod_config(neuropod_path)
    platform = neuropod_config["platform"]

    if platform == "python":
        from neuropods.backends.python.executor import PythonNeuropodExecutor
        return PythonNeuropodExecutor(neuropod_path)
    elif platform == "torchscript":
        from neuropods.backends.torchscript.executor import TorchScriptNeuropodExecutor
        return TorchScriptNeuropodExecutor(neuropod_path)
    else:
        raise ValueError("Invalid platform found in neuropod config: {}".format(platform))


if __name__ == '__main__':
    import argparse
    import cPickle as pickle

    parser = argparse.ArgumentParser(description='Load a model and run inference with provided sample data.')
    parser.add_argument('--neuropod-path', help='The path to a neuropod to load', required=True)
    parser.add_argument('--input-pkl-path', help='The path to sample input data for the model', required=True)
    parser.add_argument('--output-pkl-path', help='Where to write the output of the model', default=None)
    args = parser.parse_args()

    with load_neuropod(args.neuropod_path) as model:
        # Load the input data
        with open(args.input_pkl_path, 'rb') as pkl:
            input_data = pickle.load(pkl)

        out = model.infer(input_data)

        # Save the output data to the requested path
        if args.output_pkl_path is not None:
            with open(args.output_pkl_path, 'wb') as pkl:
                pickle.dump(out, pkl)
