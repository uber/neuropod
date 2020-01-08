# This image builds Neuropod and runs the Python and C++ tests

# To only install deps (and skip build and tests), pass `--target neuropod-base` to docker build
# FROM ubuntu:16.04 as neuropod-base
ARG NEUROPOD_CUDA_VERSION=10.0
FROM nvidia/cuda:${NEUROPOD_CUDA_VERSION}-cudnn7-runtime-ubuntu16.04 as neuropod-base

# We use sudo in the build scripts
RUN apt-get update && apt-get install -y sudo

# Create folder structure needed for installing dependencies
RUN mkdir -p /usr/src/build
WORKDIR /usr/src
COPY build/install_system_deps.sh /usr/src/build/install_system_deps.sh

# Should be set to `python` or `python3`
ARG NEUROPOD_PYTHON_BINARY=python
ENV NEUROPOD_PYTHON_BINARY=$NEUROPOD_PYTHON_BINARY

# Install system dependencies
RUN /usr/src/build/install_system_deps.sh

# Prefetch llvm because it's large
COPY build/workspace_prefetch /usr/src/source/WORKSPACE
COPY source/bazel/toolchain.patch /usr/src/source/bazel/toolchain.patch
COPY source/bazel/BUILD /usr/src/source/bazel/BUILD
RUN cd source && bazel sync

# Do everything in a virtualenv
ENV VIRTUAL_ENV=/tmp/neuropod_venv
RUN ${NEUROPOD_PYTHON_BINARY} -m pip install virtualenv && \
    ${NEUROPOD_PYTHON_BINARY} -m virtualenv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the python code into the image
RUN mkdir -p /usr/src/source/python /usr/src/source/neuropod/python
COPY build/install_python_deps.sh /usr/src/build/install_python_deps.sh
COPY build/install_frameworks.py /usr/src/build/install_frameworks.py
COPY source/python /usr/src/source/python

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

# Enable code coverage
ENV LLVM_PROFILE_FILE="/tmp/neuropod_coverage/code-%p-%9m.profraw"
ENV COVERAGE_PROCESS_START="/usr/src/source/python/.coveragerc"
RUN echo "import coverage; coverage.process_startup()" > \
    `python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`/coverage.pth

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
