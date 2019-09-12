#!/bin/bash
set -e
pushd source

# Set LD_LIBRARY_PATH
source ../build/set_build_env.sh

# Run python tests
pushd python
python -m unittest discover --verbose neuropods/tests

# Test the native bindings
NEUROPODS_RUN_NATIVE_TESTS=true python -m unittest discover --verbose neuropods/tests

# Run CPU only python tests
python -m unittest discover --verbose neuropods/tests -p cpu_test*.py

# Run CPU only python tests with native bindings
NEUROPODS_RUN_NATIVE_TESTS=true python -m unittest discover --verbose neuropods/tests -p cpu_test*.py
popd

# Run native tests
python ../build/run_cpp_tests.py
popd
