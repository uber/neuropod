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

pushd source

# Add the python library to the pythonpath
export PYTHONPATH=$PYTHONPATH:`pwd`/python

if [[ $(uname -s) == 'Linux' ]]; then
    # On linux we don't want to use GCC5 to build the custom ops
    export TF_CXX=g++-4.8
else
    # For building custom ops
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-10.15}"
fi

# Run python tests
pushd python
NEUROPOD_LOG_LEVEL=TRACE python -m unittest discover --verbose neuropod
popd

# Run native and java tests
export PATH=$PATH:`pwd`/bazel-bin/neuropod/multiprocess/

if [ -z ${NEUROPOD_TEST_FRAMEWORKS+x} ]; then
    # Run all tests if NEUROPOD_TEST_FRAMEWORKS is unset
    TEST_TARGETS=$(bazel query "kind(_test, //...)")
else
    # Get all test targets that match our framework filters
    # All tests - tests that require a framework + tests that require any of the available frameworks
    # TODO(vip): this doesn't currently handle tests that require multiple frameworks, but we don't have any of those at the moment
    TEST_TARGETS=$(bazel query "kind(_test, //...) - attr(tags, '\\brequires_framework_', //...) + attr(tags, '\\brequires_framework_(${NEUROPOD_TEST_FRAMEWORKS//,/|})\\b', //...)")
fi

# CPU tests with trace logging
bazel test "$@" --sandbox_writable_path="$HOME/.neuropod/pythonpackages/" --test_lang_filters="-java" --test_tag_filters="-gpu,-no_trace_logging" --test_env="NEUROPOD_LOG_LEVEL=TRACE" $TEST_TARGETS

# CPU tests without trace logging
bazel test "$@" --sandbox_writable_path="$HOME/.neuropod/pythonpackages/" --test_lang_filters="-java" --test_tag_filters="-gpu,no_trace_logging" $TEST_TARGETS

# Java CPU tests
bazel test "$@" --sandbox_writable_path="$HOME/.neuropod/pythonpackages/" --combined_report=lcov --test_lang_filters="java" --test_tag_filters="-gpu" $TEST_TARGETS

popd

# Maybe upload a release
python build/upload_release.py
