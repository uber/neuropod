#!/bin/bash
set -e

# Default to python 2 if not set
NEUROPODS_PYTHON_BINARY="${NEUROPODS_PYTHON_BINARY:-python}"

if [[ $(uname -s) == 'Darwin' ]]; then
    # Install bazel
    export HOMEBREW_NO_AUTO_UPDATE=1
    brew tap bazelbuild/tap
    brew install bazelbuild/tap/bazel
    brew install libomp
else
    # Install pip and bazel dependencies
    sudo apt-get update
    sudo apt-get install -y openjdk-8-jdk curl wget

    # Add bazel sources
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

    # Install bazel and python dev
    sudo apt-get update
    sudo apt-get install -y bazel ${NEUROPODS_PYTHON_BINARY}-dev ${NEUROPODS_PYTHON_BINARY}-pip

    # Install g++-4.9 and make it default to support PyTorch and TensorFlow custom ops
    sudo apt-get install -y software-properties-common g++-4.9
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 100
    sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 100
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 100
    sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 100
fi

# Run a bazel command to extract the bazel installation
bazel version
