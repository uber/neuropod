#!/bin/bash
set -e
pushd source

# Add the python library to the pythonpath
export PYTHONPATH=$PYTHONPATH:`pwd`/python

# On linux we don't want to use GCC5 to build the custom ops
if [[ $(uname -s) == 'Linux' ]]; then
    export TF_CXX=g++-4.8
fi

# Run python tests
pushd python
python -m unittest discover --verbose neuropod/tests

# Test the native bindings
NEUROPOD_RUN_NATIVE_TESTS=true python -m unittest discover --verbose neuropod/tests
popd

# Run native tests
python ../build/run_cpp_tests.py
popd

# Maybe upload a release
python build/upload_release.py
