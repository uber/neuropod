#!/bin/bash
set -e

# We need a newer version of libstdc++ for infer to run
# We also need python3.8 for black to run
apt-get install -y --no-install-recommends software-properties-common
add-apt-repository -y ppa:ubuntu-toolchain-r/test
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libstdc++6 python3.8 python3.8-dev tzdata jq
apt-get install -y git

# Install infer
VERSION=0.17.0; \
curl -sSL "https://github.com/facebook/infer/releases/download/v$VERSION/infer-linux64-v$VERSION.tar.xz" \
| sudo tar -C /opt -xJ && \
ln -s "/opt/infer-linux64-v$VERSION/bin/infer" /usr/local/bin/infer

# Get run-clang-format
wget https://raw.githubusercontent.com/Sarcasm/run-clang-format/de6e8ca07d171a7f378d379ff252a00f2905e81d/run-clang-format.py

# Install pip, black, and flake8
wget https://bootstrap.pypa.io/get-pip.py
python3.8 get-pip.py
# pin click to 8.0.4 since default click version 8.1.0 is not compatible with black version 19.10b0
python3.8 -m pip install click==8.0.4
python3.8 -m pip install black==19.10b0 flake8

# Get buildifier (for linting bazel files)
wget https://github.com/bazelbuild/buildtools/releases/download/2.2.1/buildifier -O /tmp/buildifier
chmod +x /tmp/buildifier
