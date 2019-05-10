FROM ubuntu:16.04

RUN apt-get update && apt-get install -y gcc-4.9 g++-4.9 cmake wget unzip python-pip libeigen3-dev libboost-python-dev python-numpy

# Use GCC 4.9 to build
ENV CC=/usr/bin/gcc-4.9 CXX=/usr/bin/g++-4.9

# Download and unpack libtorch and tensorflow
RUN mkdir -p /usr/deps
WORKDIR /usr/deps
RUN wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-1.0.0.dev20190318.zip && \
    unzip libtorch-*.zip && \
    mkdir tensorflow && \
    wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz && \
    tar -xvf libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz -C tensorflow

# Download and build jsoncpp
RUN wget https://github.com/open-source-parsers/jsoncpp/archive/0.8.0.tar.gz && \
    tar -xvf 0.8.0.tar.gz && \
    mkdir -p /usr/deps/jsoncpp-0.8.0/build && \
    cd /usr/deps/jsoncpp-0.8.0/build && \
    cmake -DBUILD_SHARED_LIBS=ON .. && \
    make && \
    make install && \
    ldconfig

# Copy the python code into the image
RUN mkdir -p /usr/src/source/python /usr/src/source/neuropods/python
COPY source/python /usr/src/source/python
COPY source/neuropods/python /usr/src/source/neuropods/python

# Install deps for the python interface
# (the -f flag tells pip where to find the torch nightly builds)
WORKDIR /usr/src/source/python
RUN pip install -U pip setuptools && \
    python setup.py egg_info && \
    cat neuropods.egg-info/requires.txt  | sed '/^\[/ d' | paste -sd " " - | xargs pip install -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# Run python tests
WORKDIR /usr/src/source/python
RUN python -m unittest discover --verbose neuropods/tests

# Build a wheel + install locally
RUN python setup.py bdist_wheel && pip install dist/*.whl

# Copy the rest of the code in
RUN mkdir -p /usr/src
COPY . /usr/src

# Build native library and run tests
WORKDIR /usr/src/source/neuropods
RUN mkdir build && \
    cd build && \
    cmake --warn-uninitialized -DCMAKE_PREFIX_PATH="/usr/deps/libtorch;/usr/deps/tensorflow;/usr/deps/jsoncpp-0.8.0" .. && \
    make && \
    cd tests && \
    ctest --output-on-failure

# Make sure we only depend on .so files we whitelist (and we depend on all the whitelisted ones)
RUN mkdir -p /tmp/whitelist_check && \
    (find build/ -name *.so | xargs -i cp {} /tmp/whitelist_check) && \
    readelf -d /tmp/whitelist_check/*.so | grep NEEDED | sort | uniq |\
    diff -I '^#.*' /usr/src/build/allowed_deps.txt -
