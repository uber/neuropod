#!/bin/bash
set -e

# We need a newer version of libstdc++ for infer to run
# We also need python3.6 for black to run
apt-get install -y --no-install-recommends software-properties-common
add-apt-repository -y ppa:ubuntu-toolchain-r/test
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y --no-install-recommends libstdc++6 python3.6 python3.6-dev tzdata jq

# Install bazel-compdb
INSTALL_DIR="/usr/local/bin"
VERSION="0.4.2"
(
  cd "${INSTALL_DIR}" \
  && curl -L "https://github.com/grailbio/bazel-compilation-database/archive/${VERSION}.tar.gz" | tar -xz \
  && ln -f -s "${INSTALL_DIR}/bazel-compilation-database-${VERSION}/generate.sh" bazel-compdb
)

# Install infer
VERSION=0.17.0; \
curl -sSL "https://github.com/facebook/infer/releases/download/v$VERSION/infer-linux64-v$VERSION.tar.xz" \
| sudo tar -C /opt -xJ && \
ln -s "/opt/infer-linux64-v$VERSION/bin/infer" /usr/local/bin/infer

# Get run-clang-format
wget https://raw.githubusercontent.com/Sarcasm/run-clang-format/de6e8ca07d171a7f378d379ff252a00f2905e81d/run-clang-format.py

# Install pip, black, and flake8
wget https://bootstrap.pypa.io/get-pip.py
python3.6 get-pip.py
python3.6 -m pip install black flake8
