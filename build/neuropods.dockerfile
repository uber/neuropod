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
