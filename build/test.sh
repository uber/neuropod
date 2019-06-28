#!/bin/bash
set -e
pushd source

# Set the build environment
source ../build/set_build_env.sh

# Run python tests
pushd python
python -m unittest discover --verbose neuropods/tests
popd

# Run native tests
bazel test --cache_test_results=no --test_output=errors "$@" //...
popd
