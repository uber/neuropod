#!/bin/bash
set -e
pushd source

# Run python tests
pushd python
python -m unittest discover --verbose neuropods/tests

# Build a wheel + install locally
python setup.py bdist_wheel && pip install dist/*.whl
popd

# Add TF and torch to the LD_LIBRARY_PATH
# This is so the tests can find the shared objects at runtime
# TODO(vip): find a better way to do this
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/bazel-source/external/libtorch_repo/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/bazel-source/external/tensorflow_repo/lib

# Make sure the tests can find all the `.so` files for the backends
# TODO(vip): get rpaths to work correctly in bazel on linux
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/dist_test/lib

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

# Run native tests
bazel test --test_output=errors //...
popd
