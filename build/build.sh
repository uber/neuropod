#!/bin/bash
set -ex

# Use the virtualenv
source .neuropod_venv/bin/activate

pushd source

# Build the native code
mkdir -p "$HOME/.neuropod/pythonpackages/"
bazel build --sandbox_writable_path="$HOME/.neuropod/pythonpackages/" "$@" //...:all //neuropod:packages

# Copy the binaries needed by the python bindings
cp bazel-bin/neuropod/bindings/neuropod_native.so python/neuropod/
cp bazel-bin/neuropod/libneuropod.so python/neuropod/
cp bazel-bin/neuropod/multiprocess/neuropod_multiprocess_worker python/neuropod/

# Build a wheel
pushd python
if [[ $(uname -s) == 'Darwin' ]]; then
    PLATFORM_TAG=`python -c 'import distutils.util;print(distutils.util.get_platform().replace("-","_").replace(".","_"))'`
else
    PLATFORM_TAG="manylinux2014_x86_64"
fi
python setup.py bdist_wheel --plat-name "$PLATFORM_TAG"
popd

# Install the backends to our test base directory
rm -rf "../.neuropod_test_base" && mkdir "../.neuropod_test_base"
tar -xf "./bazel-bin/neuropod/backends/tensorflow/neuropod_tensorflow_backend.tar.gz" -C "../.neuropod_test_base"
tar -xf "./bazel-bin/neuropod/backends/torchscript/neuropod_torchscript_backend.tar.gz" -C "../.neuropod_test_base"
tar -xf "./bazel-bin/neuropod/backends/python_bridge/neuropod_pythonbridge_backend.tar.gz" -C "../.neuropod_test_base"

# Add the python libray to the pythonpath
export PYTHONPATH=$PYTHONPATH:`pwd`/python

if [[ $(uname -s) == 'Linux' ]]; then
    # Copy the build artificts into a dist folder
    mkdir -p /tmp/neuropod_dist && \
        cp bazel-bin/neuropod/libneuropod.tar.gz python/dist/*.whl /tmp/neuropod_dist

    # Make sure we only depend on .so files we whitelist (and we depend on all the whitelisted ones)
    # Depending on the version of torch, the dependency is either `libtorch.so` or `libtorch.so.1`.
    # Similarly for tensorflow.
    # Because of this, we don't include them in our dependency check
    mkdir -p /tmp/dist_test && \
        tar -xvf /tmp/neuropod_dist/libneuropod.tar.gz -C /tmp/dist_test && \
        readelf -d /tmp/dist_test/lib/*.so | grep NEEDED | sort | uniq |\
        grep -v libtorch.so |\
        grep -v libtensorflow.so |\
        grep -v libpython |\
        diff -I '^#.*' ../build/allowed_deps.txt -
fi

popd
