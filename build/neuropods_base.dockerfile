# This is the base image for building Neuropods. It installs all the dependencies
# (bazel, pip, etc.) so that the "real" build is faster.
# This image should be rebuilt every time setup.py or system dependencies change
#
# docker build -f build/neuropods_base.dockerfile -t neuropods_base .

FROM ubuntu:16.04

# Install pip and bazel dependencies
RUN apt-get update && apt-get install -y openjdk-8-jdk curl wget gcc-4.9 g++-4.9 python-pip

# Add bazel sources
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
  && curl https://bazel.build/bazel-release.pub.gpg | apt-key add -

# Install bazel and other native deps
RUN apt-get update && \
    apt-get install -y bazel python-dev libboost-python-dev && \
    rm -rf /var/lib/apt/lists/*

# Run a bazel command to extract the bazel installation
RUN bazel version

# Download and install libtorch
RUN mkdir -p /usr/deps && \
    wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip && \
    unzip -d /usr/deps libtorch-shared-with-deps-latest.zip && \
    echo "/usr/deps/libtorch/lib" > /etc/ld.so.conf.d/libtorch.conf && \
    ldconfig

# Create a source dir and copy the code in
RUN mkdir -p /usr/src
COPY . /usr/src

# Make sure we build with gcc/g++ 4.9
# (to make sure that libtorch works)
ENV CC=/usr/bin/gcc-4.9 CXX=/usr/bin/g++-4.9

# Install deps for the python interface
WORKDIR /usr/src/source/python
RUN pip install -U setuptools && \
    python setup.py egg_info && \
    cat neuropods.egg-info/requires.txt  | sed '/^\[/ d' | paste -sd " " - | xargs pip install

# Delete the source code we copied in
WORKDIR /usr/src
RUN rm -rf /usr/src/*
