#!/bin/bash
set -e
pushd source

# Make sure we have GPUs
num_gpus=`nvidia-smi -L | wc -l`
if [ "$num_gpus" == "0" ]; then
    echo "Attemped to run tests on GPU, but no GPUs found!"
    exit 1
else
    echo "Found GPUs!"
    # List the GPUs
    nvidia-smi -L
fi

# Set LD_LIBRARY_PATH
source ../build/set_build_env.sh

# Run python tests
pushd python
python -m unittest discover --verbose neuropods/tests

# Test the native bindings
NEUROPODS_RUN_NATIVE_TESTS=true python -m unittest discover --verbose neuropods/tests

# Run GPU only python tests
python -m unittest discover --verbose neuropods/tests -p gpu_test*.py

# Run GPU only python tests with native bindings
NEUROPODS_RUN_NATIVE_TESTS=true python -m unittest discover --verbose neuropods/tests -p gpu_test*.py
popd

# Run native tests
bazel test --cache_test_results=no --test_output=errors "$@" //...
popd
