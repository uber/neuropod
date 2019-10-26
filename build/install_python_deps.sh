#!/bin/bash
set -e

# Install deps for the python interface
# (the -f flag tells pip where to find the torch nightly builds)
pushd source/python
pip install -U pip setuptools numpy coverage requests mkdocs mkdocs-material
python setup.py egg_info
cat neuropods.egg-info/requires.txt | sed '/^\[/ d' | paste -sd " " - | xargs pip install
popd

# Install the appropriate versions of torch and TF
python ./build/install_frameworks.py
