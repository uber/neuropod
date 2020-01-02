#!/bin/bash
set -e
pushd source

# Set LD_LIBRARY_PATH
source ../build/set_build_env.sh

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
