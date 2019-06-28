# This image builds Neuropods and runs the Python and C++ tests

# To only install deps (and skip build and tests), pass `--target neuropod-base` to docker build
# FROM ubuntu:16.04 as neuropod-base
FROM nvidia/cuda:10.0-runtime-ubuntu16.04 as neuropod-base

# Optional overrides used by the bazel build
ARG NEUROPODS_TENSORFLOW_VERSION
ARG NEUROPODS_TENSORFLOW_URL
ARG NEUROPODS_TENSORFLOW_SHA256
ARG NEUROPODS_LIBTORCH_VERSION
ARG NEUROPODS_LIBTORCH_URL
ARG NEUROPODS_LIBTORCH_SHA256

ENV NEUROPODS_TENSORFLOW_VERSION=$NEUROPODS_TENSORFLOW_VERSION
ENV NEUROPODS_TENSORFLOW_URL=$NEUROPODS_TENSORFLOW_URL
ENV NEUROPODS_TENSORFLOW_SHA256=$NEUROPODS_TENSORFLOW_SHA256
ENV NEUROPODS_LIBTORCH_VERSION=$NEUROPODS_LIBTORCH_VERSION
ENV NEUROPODS_LIBTORCH_URL=$NEUROPODS_LIBTORCH_URL
ENV NEUROPODS_LIBTORCH_SHA256=$NEUROPODS_LIBTORCH_SHA256

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

# Build
# To only build (and skip tests), pass `--target neuropod-build` to docker build
FROM neuropod-base as neuropod-build
RUN /usr/src/build/build.sh

# Test
FROM neuropod-build
RUN /usr/src/build/test.sh
