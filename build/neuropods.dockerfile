# This image builds Neuropods and runs the python and C++ tests
# It starts from the `neuropods_base` image which contains some of the large
# dependencies for Neuropods

# TODO(vip): After open sourcing, update the base image (and use `neuropods_base.dockerfile` to generate it)
# For now, this public base image has all the deps installed (but has no references to neuropods)
# FROM neuropods_base
FROM vpanyam/dl_base:latest

# Set the GCC version used for this build
ARG GCC_VERSION=4.9
ENV CC=/usr/bin/gcc-${GCC_VERSION} CXX=/usr/bin/g++-${GCC_VERSION}

# Optional overrides used by the bazel build
ARG NEUROPODS_TENSORFLOW_VERSION
ARG NEUROPODS_TENSORFLOW_URL
ARG NEUROPODS_TENSORFLOW_SHA256
ARG NEUROPODS_PYTORCH_VERSION
ARG NEUROPODS_PYTORCH_URL
ARG NEUROPODS_PYTORCH_SHA256

ENV NEUROPODS_TENSORFLOW_VERSION=$NEUROPODS_TENSORFLOW_VERSION
ENV NEUROPODS_TENSORFLOW_URL=$NEUROPODS_TENSORFLOW_URL
ENV NEUROPODS_TENSORFLOW_SHA256=$NEUROPODS_TENSORFLOW_SHA256
ENV NEUROPODS_PYTORCH_VERSION=$NEUROPODS_PYTORCH_VERSION
ENV NEUROPODS_PYTORCH_URL=$NEUROPODS_PYTORCH_URL
ENV NEUROPODS_PYTORCH_SHA256=$NEUROPODS_PYTORCH_SHA256

# Install any pip packages that were requested
# This lets us do things like using a different build of torch
# TODO(vip): Move this into bazel
ARG PIP_OVERRIDES
RUN [ ! -z "${PIP_OVERRIDES}" ] && pip install ${PIP_OVERRIDES} || echo "No pip overrides specified."

# TODO(vip): for some reason, adding to `/etc/ld.so.conf.d/libtorch.conf` does not work for
# new versions of libtorch. Setting LD_LIBRARY_PATH does work correctly, however
ENV LD_LIBRARY_PATH="/usr/src/source/bazel-source/external/libtorch_repo/lib"

# Create a source dir and copy the code in
RUN mkdir -p /usr/src
COPY . /usr/src

# Run python tests
WORKDIR /usr/src/source/python
RUN python -m unittest discover --verbose neuropods/tests

# Build a wheel + install locally
RUN python setup.py bdist_wheel && pip install dist/*.whl

# Update PATH so the tests can find binaries they need
ENV PATH="/tmp/dist_test/bin:${PATH}"

# Build the native code
WORKDIR /usr/src/source
RUN bazel build //...:all

# Copy the build artificts into a dist folder
RUN mkdir -p /usr/src/dist && \
    cp bazel-bin/neuropods/libneuropods.tar.gz python/dist/*.whl /usr/src/dist

# Make sure we only depend on .so files we whitelist (and we depend on all the whitelisted ones)
RUN mkdir -p /tmp/dist_test && \
    tar -xvf /usr/src/dist/libneuropods.tar.gz -C /tmp/dist_test && \
    readelf -d /tmp/dist_test/lib/* /tmp/dist_test/bin/* | grep NEEDED | sort | uniq |\
    diff -I '^#.*' /usr/src/build/allowed_deps.txt -

# Make sure the tests can find all the `.so` files for the backends
RUN echo "/tmp/dist_test/lib" > /etc/ld.so.conf.d/libneuropods.conf && \
    echo "/usr/src/source/bazel-source/external/tensorflow_repo/lib" > /etc/ld.so.conf.d/tensorflow.conf && \
    ldconfig

# Run native tests
RUN bazel test --test_output=errors //...
