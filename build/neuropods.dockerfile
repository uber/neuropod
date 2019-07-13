# This image builds Neuropods and runs the Python and C++ tests

# To only install deps (and skip build and tests), pass `--target neuropod-base` to docker build
# FROM ubuntu:16.04 as neuropod-base
ARG NEUROPODS_CUDA_VERSION=10.0
FROM nvidia/cuda:${NEUROPODS_CUDA_VERSION}-cudnn7-runtime-ubuntu16.04 as neuropod-base

# We use sudo in the build scripts
RUN apt-get update && apt-get install -y sudo

# Create folder structure needed for installing dependencies
RUN mkdir -p /usr/src/build
WORKDIR /usr/src
COPY build/install_system_deps.sh /usr/src/build/install_system_deps.sh

# Should be set to `python` or `python3`
ARG NEUROPODS_PYTHON_BINARY
ENV NEUROPODS_PYTHON_BINARY=$NEUROPODS_PYTHON_BINARY

# Install system dependencies
RUN /usr/src/build/install_system_deps.sh

# Do everything in a virtualenv
ENV VIRTUAL_ENV=/tmp/neuropod_venv
RUN ${NEUROPODS_PYTHON_BINARY} -m pip install virtualenv && \
    ${NEUROPODS_PYTHON_BINARY} -m virtualenv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the python code into the image
RUN mkdir -p /usr/src/source/python /usr/src/source/neuropods/python
COPY build/install_python_deps.sh /usr/src/build/install_python_deps.sh
COPY build/install_frameworks.py /usr/src/build/install_frameworks.py
COPY source/python /usr/src/source/python

# Optional overrides
ARG NEUROPODS_TENSORFLOW_VERSION
ARG NEUROPODS_TORCH_VERSION
ARG NEUROPODS_IS_GPU
ARG NEUROPODS_CUDA_VERSION

ENV NEUROPODS_TENSORFLOW_VERSION=$NEUROPODS_TENSORFLOW_VERSION
ENV NEUROPODS_TORCH_VERSION=$NEUROPODS_TORCH_VERSION
ENV NEUROPODS_IS_GPU=$NEUROPODS_IS_GPU
ENV NEUROPODS_CUDA_VERSION=$NEUROPODS_CUDA_VERSION

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
