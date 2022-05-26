#!/bin/bash
set -e

# Defaults to python 2 if not set
NEUROPOD_PYTHON_BINARY="python${NEUROPOD_PYTHON_VERSION}"

# Install pip
wget https://bootstrap.pypa.io/pip/3.6/get-pip.py -O /tmp/get-pip.py
${NEUROPOD_PYTHON_BINARY} /tmp/get-pip.py

# Setup a virtualenv
${NEUROPOD_PYTHON_BINARY} -m pip install virtualenv==16.7.9
${NEUROPOD_PYTHON_BINARY} -m virtualenv .neuropod_venv
source .neuropod_venv/bin/activate

# Install deps for the python interface and tests
pushd source/python
pip install -U pip setuptools numpy coverage requests[security] protobuf==3.19.4
python setup.py egg_info
cat neuropod.egg-info/requires.txt | sed '/^\[/ d' | paste -sd " " - | xargs pip install
popd

# Install the appropriate versions of torch and TF
python ./build/install_frameworks.py
