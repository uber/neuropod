#!/bin/bash
set -e

if [[ $(uname -s) == 'Darwin' ]]; then

    # Get bazelisk if necessary
    if [ ! -f "/usr/local/bin/bazel" ]; then
        wget https://github.com/bazelbuild/bazelisk/releases/download/v1.4.0/bazelisk-darwin-amd64 -O /usr/local/bin/bazel
        chmod +x /usr/local/bin/bazel
    fi

    # Install libomp
    export HOMEBREW_NO_AUTO_UPDATE=1
    brew install libomp
else
    # Install bazel deps
    # Install g++-4.8 for TensorFlow custom op builds
    sudo apt-get update
    sudo apt-get install -y \
        pkg-config \
        zip \
        g++ \
        zlib1g-dev \
        unzip \
        curl \
        wget \
        g++-4.8 \
        openjdk-8-jdk \
        lcov

    # Add a repo that includes newer python versions
    sudo apt-get install -y --no-install-recommends software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update

    # Get bazelisk if necessary
    if [ ! -f "/usr/local/bin/bazel" ]; then
        tmpdir=$(mktemp -d)
        pushd $tmpdir
        wget https://github.com/bazelbuild/bazelisk/releases/download/v1.4.0/bazelisk-linux-amd64 -O ./bazel
        chmod +x ./bazel
        sudo mv ./bazel /usr/local/bin/bazel
        popd
        rm -rf $tmpdir
    fi
fi
