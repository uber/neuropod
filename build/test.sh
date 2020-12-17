#!/bin/bash
set -ex

# Use the virtualenv
source .neuropod_venv/bin/activate

# Enable code coverage
export LLVM_PROFILE_FILE="/tmp/neuropod_coverage/code-%p-%9m.profraw"
export COVERAGE_PROCESS_START="`pwd`/source/python/.coveragerc"
echo "import coverage; coverage.process_startup()" > \
    `python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`/coverage.pth

# Override the Neuropod backend base directory
export NEUROPOD_BASE_DIR=`pwd`/.neuropod_test_base

# Enable python isolation
# TODO(vip): Remove this once isolation is default behavior
export NEUROPOD_ENABLE_PYTHON_ISOLATION=true

pushd source

# Add the python library to the pythonpath
export PYTHONPATH=$PYTHONPATH:`pwd`/python

if [[ $(uname -s) == 'Linux' ]]; then
    # On linux we don't want to use GCC5 to build the custom ops
    export TF_CXX=g++-4.8
else
    # For building custom ops
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-10.13}"
fi

# Run python tests
pushd python
NEUROPOD_LOG_LEVEL=TRACE python -m unittest discover --verbose neuropod/tests

# Test the native bindings
NEUROPOD_LOG_LEVEL=TRACE NEUROPOD_RUN_NATIVE_TESTS=true python -m unittest discover --verbose neuropod/tests
popd

# Run native and java tests
export PATH=$PATH:`pwd`/bazel-bin/neuropod/multiprocess/

# CPU tests with trace logging
bazel test "$@" --sandbox_writable_path="$HOME/.neuropod/pythonpackages/" --test_lang_filters="-java" --test_tag_filters="-gpu,-no_trace_logging" --test_env="NEUROPOD_LOG_LEVEL=TRACE" //...

# CPU tests without trace logging
bazel test "$@" --sandbox_writable_path="$HOME/.neuropod/pythonpackages/" --test_lang_filters="-java" --test_tag_filters="-gpu,no_trace_logging" //...

# Java CPU tests
bazel test "$@" --sandbox_writable_path="$HOME/.neuropod/pythonpackages/" --combined_report=lcov --test_lang_filters="java" --test_tag_filters="-gpu" //...

popd

# Maybe upload a release
python build/upload_release.py
