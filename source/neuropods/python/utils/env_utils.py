#
# Uber, Inc. (c) 2018
#

"""
Utilities for dealing with virtualenvs and pip
"""
import cPickle as pickle
import os
import shutil
import subprocess

from tempfile import mkstemp
from testpath.tempdir import TemporaryDirectory

import neuropods


def create_virtualenv(venv_path, packages_to_install=[], verbose=False):
    with open(os.devnull, 'w') as FNULL:
        stdout = None if verbose else FNULL

        # Create the virtualenv
        subprocess.call(['/usr/bin/env', 'virtualenv', venv_path], env={}, stdout=stdout)

        # Install the specified pip packages
        for package in ['pip', 'six'] + packages_to_install:
            subprocess.call([os.path.join(venv_path, 'bin', 'pip'), 'install', '-U', package], env={}, stdout=stdout)

        # Copy the neuropods library into the virtualenv
        # This dir is added to the pythonpath in `eval_in_virtualenv`
        shutil.copytree(neuropods.__path__[0], os.path.join(venv_path, "neuropod_test_libs", "neuropods"))


def eval_in_virtualenv(neuropod_path, input_data, venv_path):
    """
    Loads and runs a neuropod model in a separate process with specified input data.
    It also does this in a virtualenv in order to make sure that the model was
    packaged correctly and does not have any unspecified dependencies.

    Raises a CalledProcessError if there was an error evaluating the neuropod

    :param  neuropod_path   The path to the neuropod to load
    :param  input_data      A pickleable dict containing sample input to the model
    :param  venv_path       The path to a virtualenv to run the model in
    """
    with TemporaryDirectory() as tmpdir:
        _, input_filepath = mkstemp(dir=tmpdir)
        with open(input_filepath, 'wb') as input_pkl:
            pickle.dump(input_data, input_pkl, protocol=-1)

        _, output_filepath = mkstemp(dir=tmpdir)

        subprocess.check_call([os.path.join(venv_path, 'bin', 'python'), '-m', 'neuropods.loader',
                               '--neuropod-path', neuropod_path,
                               '--input-pkl-path', input_filepath,
                               '--output-pkl-path', output_filepath,
                               ], env={"PYTHONPATH": os.path.join(venv_path, "neuropod_test_libs")})

        with open(output_filepath, 'rb') as output_pkl:
            return pickle.load(output_pkl)


if __name__ == '__main__':
    create_virtualenv("/tmp/neuropod_test_venv", packages_to_install=['torch', 'torchvision'])
