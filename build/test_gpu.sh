#!/bin/bash
set -e

# Use the virtualenv
source .neuropod_venv/bin/activate

# Enable code coverage
export LLVM_PROFILE_FILE="/tmp/neuropod_coverage/code-%p-%9m.profraw"
export COVERAGE_PROCESS_START="`pwd`/source/python/.coveragerc"
echo "import coverage; coverage.process_startup()" > \
    `python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`/coverage.pth

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

# Run GPU only python tests
python -m unittest discover --verbose neuropod/tests -p gpu_test*.py

# Run GPU only python tests with native bindings
NEUROPOD_RUN_NATIVE_TESTS=true python -m unittest discover --verbose neuropod/tests -p gpu_test*.py
popd

# Run native tests
python ../build/run_cpp_tests.py --run-gpu-tests
popd

# Maybe upload a release
python build/upload_release.py
