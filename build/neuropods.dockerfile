# This image builds Neuropods and runs the python and C++ tests
# It starts from the `neuropods_base` image which contains some of the large
# dependencies for Neuropods

FROM neuropods_base

# Create a source dir and copy the code in
RUN mkdir -p /usr/src
COPY . /usr/src

# Run python tests
WORKDIR /usr/src/source/python
RUN python -m unittest discover --verbose neuropods/tests

# Build a wheel + install locally
RUN python setup.py bdist_wheel && pip install dist/*.whl

# Build the native code
WORKDIR /usr/src/source
RUN bazel build //...:all

# Run native tests
RUN bazel test //...

# Copy the build artificts into a dist folder
RUN mkdir -p /usr/src/dist && \
    cp bazel-bin/neuropods/libneuropods.tar.gz python/dist/*.whl /usr/src/dist

# Make sure we only depend on .so files we whitelist (and we depend on all the whitelisted ones)
RUN mkdir -p /tmp/dist_test && \
    tar -xvf /usr/src/dist/libneuropods.tar.gz -C /tmp/dist_test && \
    readelf -d /tmp/dist_test/lib/*.so | grep NEEDED | sort | uniq |\
    diff -I '^#.*' /usr/src/build/allowed_deps.txt -
