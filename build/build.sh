#!/bin/bash
set -e
pushd source

# Set LD_LIBRARY_PATH
source ../build/set_build_env.sh

# Build a wheel
pushd python
python setup.py bdist_wheel
popd

# Build the native code
bazel build "$@" //...:all

# Build the packages if we need to (at least one of these are not empty)
if [[ ! -z "${NEUROPODS_DO_PACKAGE}${BUILDKITE_TAG}${TRAVIS_TAG}" ]]; then
    bazel build "$@" //neuropods:packages
fi

if [[ $(uname -s) == 'Linux' ]]; then
    # Copy the build artificts into a dist folder
    mkdir -p /tmp/neuropod_dist && \
        cp bazel-bin/neuropods/libneuropods.tar.gz python/dist/*.whl /tmp/neuropod_dist

    # Make sure we only depend on .so files we whitelist (and we depend on all the whitelisted ones)
    # Depending on the version of torch, the dependency is either `libtorch.so` or `libtorch.so.1`.
    # Similarly for tensorflow.
    # Because of this, we don't include them in our dependency check
    mkdir -p /tmp/dist_test && \
        tar -xvf /tmp/neuropod_dist/libneuropods.tar.gz -C /tmp/dist_test && \
        readelf -d /tmp/dist_test/lib/*.so | grep NEEDED | sort | uniq |\
        grep -v libtorch.so |\
        grep -v libtensorflow.so |\
        grep -v libpython |\
        diff -I '^#.*' ../build/allowed_deps.txt -
fi

popd
