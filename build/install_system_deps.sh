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
fi

# Run a bazel command to extract the bazel installation
bazel version
