#!/bin/bash
set -e
pushd source

# Set LD_LIBRARY_PATH
source ../build/set_build_env.sh

# Build a wheel + install locally
pushd python
python setup.py bdist_wheel && pip install dist/*.whl
popd

# Build the native code
bazel build //...:all

if [[ $(uname -s) == 'Linux' ]]; then
    # Copy the build artificts into a dist folder
    mkdir -p /tmp/neuropod_dist && \
        cp bazel-bin/neuropods/libneuropods.tar.gz python/dist/*.whl /tmp/neuropod_dist

    # Make sure we only depend on .so files we whitelist (and we depend on all the whitelisted ones)
    mkdir -p /tmp/dist_test && \
        tar -xvf /tmp/neuropod_dist/libneuropods.tar.gz -C /tmp/dist_test && \
        readelf -d /tmp/dist_test/lib/*.so | grep NEEDED | sort | uniq |\
        diff -I '^#.*' ../build/allowed_deps.txt -
fi

popd
