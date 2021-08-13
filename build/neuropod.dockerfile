# This image builds Neuropod and runs the Python and C++ tests

# To only install deps (and skip build and tests), pass `--target neuropod-base` to docker build
# FROM ubuntu:16.04 as neuropod-base
ARG NEUROPOD_CUDA_VERSION=10.0
ARG NEUROPOD_CUDNN_VERSION=7
FROM nvidia/cuda:${NEUROPOD_CUDA_VERSION}-cudnn${NEUROPOD_CUDNN_VERSION}-runtime-ubuntu16.04 as neuropod-base

# Use utf8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# We use sudo in the build scripts
RUN apt-get update && apt-get install -y sudo

# Create folder structure needed for installing dependencies
RUN mkdir -p /usr/src/build
WORKDIR /usr/src
COPY build/install_system_deps.sh /usr/src/build/install_system_deps.sh

# Install system dependencies
RUN /usr/src/build/install_system_deps.sh

# Prefetch llvm because it's large
COPY build/workspace_prefetch /usr/src/source/WORKSPACE
COPY source/.bazelversion /usr/src/source/.bazelversion
COPY source/bazel/toolchain.patch /usr/src/source/bazel/toolchain.patch
COPY source/bazel/BUILD /usr/src/source/bazel/BUILD
RUN cd source && bazel sync

# The python version to use. Should be set to `2.7`, `3.5`, etc.
ARG NEUROPOD_PYTHON_VERSION=2.7
ENV NEUROPOD_PYTHON_VERSION=$NEUROPOD_PYTHON_VERSION

# Install python
RUN sudo apt-get update && \
    sudo apt-get install -y "python${NEUROPOD_PYTHON_VERSION}" "python${NEUROPOD_PYTHON_VERSION}-dev" && \
    ln -s "$(which python3)" /usr/bin/python

# For python 3.8, we need to install distutils
RUN if [ "${NEUROPOD_PYTHON_VERSION}" = "3.8" ] ; then sudo apt-get install -y "python${NEUROPOD_PYTHON_VERSION}-distutils" ; fi

# Copy the python code into the image
RUN mkdir -p /usr/src/source/python /usr/src/source/neuropod/python
COPY build/install_python_deps.sh /usr/src/build/install_python_deps.sh
COPY build/install_frameworks.py /usr/src/build/install_frameworks.py
COPY source/python/setup.py /usr/src/source/python/setup.py

# Optional overrides
ARG NEUROPOD_TENSORFLOW_VERSION
ARG NEUROPOD_TORCH_VERSION
ARG NEUROPOD_IS_GPU
ARG NEUROPOD_CUDA_VERSION

ENV NEUROPOD_TENSORFLOW_VERSION=$NEUROPOD_TENSORFLOW_VERSION
ENV NEUROPOD_TORCH_VERSION=$NEUROPOD_TORCH_VERSION
ENV NEUROPOD_IS_GPU=$NEUROPOD_IS_GPU
ENV NEUROPOD_CUDA_VERSION=$NEUROPOD_CUDA_VERSION

# Install python dependencies
RUN /usr/src/build/install_python_deps.sh

# Copy the rest of the code in
COPY . /usr/src

# Whether we should generate release packages
# We don't do this by default because packaging torch and TF can take a bit of time
# (packages are built if this is not empty)
ARG NEUROPOD_DO_PACKAGE
ENV NEUROPOD_DO_PACKAGE=$NEUROPOD_DO_PACKAGE

# Build
# To only build (and skip tests), pass `--target neuropod-build` to docker build
FROM neuropod-base as neuropod-build
RUN /usr/src/build/build.sh

# Test
FROM neuropod-build
RUN /usr/src/build/test.sh
