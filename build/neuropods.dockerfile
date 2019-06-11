# This image builds Neuropods and runs the python and C++ tests

FROM ubuntu:16.04

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

# We use sudo in the build scripts
RUN apt-get update && apt-get install -y sudo

# Create folder structure needed for installing dependencies
RUN mkdir -p /usr/src/build
WORKDIR /usr/src
COPY build/install_system_deps.sh /usr/src/build/install_system_deps.sh

# Install system dependencies
RUN /usr/src/build/install_system_deps.sh

# Copy the python code into the image
RUN mkdir -p /usr/src/source/python /usr/src/source/neuropods/python
COPY build/install_python_deps.sh /usr/src/build/install_python_deps.sh
COPY source/python /usr/src/source/python

# Install python dependencies
RUN /usr/src/build/install_python_deps.sh

# Copy the rest of the code in
COPY . /usr/src

# Build and test
# See build/docker_build.sh
