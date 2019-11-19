#
# Uber, Inc. (c) 2018
#

from six.moves import cPickle as pickle
import os
import shutil
import subprocess
import sys

from tempfile import mkstemp
from testpath.tempdir import TemporaryDirectory

import neuropods


def eval_in_new_process(neuropod_path, input_data, binary_path=sys.executable, extra_args=[], neuropod_load_args={}, **kwargs):
    """
    Loads and runs a neuropod model in a separate process with specified input data.

    Raises a CalledProcessError if there was an error evaluating the neuropod

    :param  neuropod_path       The path to the neuropod to load
    :param  input_data          A pickleable dict containing sample input to the model
    :param  binary_path         The binary to run. Defaults to "python"
    :param  extra_args          Optional extra command line args to provide
    :param  neuropod_load_args  A pickleable dict containing args to provide to `load_neuropod`
    """
    with TemporaryDirectory() as tmpdir:
        _, input_filepath = mkstemp(dir=tmpdir)
        with open(input_filepath, 'wb') as input_pkl:
            pickle.dump(input_data, input_pkl, protocol=-1)

        _, args_filepath = mkstemp(dir=tmpdir)
        with open(args_filepath, 'wb') as args_pkl:
            pickle.dump(neuropod_load_args, args_pkl, protocol=-1)

        _, output_filepath = mkstemp(dir=tmpdir)

        subprocess.check_call([binary_path, '-m', 'neuropods.loader',
                               '--neuropod-path', neuropod_path,
                               '--input-pkl-path', input_filepath,
                               '--output-pkl-path', output_filepath,
                               '--args-pkl-path', args_filepath,
                               ] + extra_args, **kwargs)

        with open(output_filepath, 'rb') as output_pkl:
            return pickle.load(output_pkl)
