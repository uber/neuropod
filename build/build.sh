#!/bin/bash
set -e

# Use the virtualenv
source .neuropod_venv/bin/activate

pushd source

# Build the native code
bazel build "$@" //...:all //neuropod:packages

# Copy the binaries needed by the python bindings
cp bazel-bin/neuropod/bindings/neuropod_native.so python/neuropod/
cp bazel-bin/neuropod/libneuropod.so python/neuropod/
cp bazel-bin/neuropod/multiprocess/neuropod_multiprocess_worker python/neuropod/

if [[ $(uname -s) == 'Darwin' ]]; then
    # Postprocessing needed on mac
    chmod 755 "python/neuropod/libneuropod.so"
    for FILE in "python/neuropod/neuropod_native.so" "python/neuropod/neuropod_multiprocess_worker"
    do
        chmod 755 $FILE
        OLD_PATH=$(otool -L ${FILE} | grep libneuropod.so | cut -d ' ' -f1 | column -t)
        install_name_tool -change "${OLD_PATH}" "@rpath/libneuropod.so" "${FILE}"
    done
fi

# Build a wheel
pushd python
if [[ $(uname -s) == 'Darwin' ]]; then
    PLATFORM_TAG=`python -c 'import distutils.util;print(distutils.util.get_platform().replace("-","_").replace(".","_"))'`
else
    PLATFORM_TAG="manylinux2014_x86_64"
fi
python setup.py bdist_wheel --plat-name "$PLATFORM_TAG"
popd

# Add the python libray to the pythonpath
export PYTHONPATH=$PYTHONPATH:`pwd`/python

# Build the wheels for the backends
python ../build/wheel/build_wheel.py

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

# Use a naive substring searching to check if coverage is requested.
if [[ $@ == *'--config=coverage'* ]]; then
    # generate coverage for Java
    pushd source
    # Set PATH for Java tests
    PATH=$PATH:`pwd`/bazel-bin/neuropod/multiprocess/
    bazel coverage --collect_code_coverage --instrumentation_filter='/java[/:]' --combined_report=lcov --coverage_report_generator=@bazel_tools//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Main //neuropod/bindings/java/...
    popd
fi
