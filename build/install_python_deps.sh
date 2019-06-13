#!/bin/bash
set -e

# Install deps for the python interface
# (the -f flag tells pip where to find the torch nightly builds)
pushd source/python
pip install -U pip setuptools numpy
python setup.py egg_info

if [[ $(uname -s) == 'Darwin' ]]; then
    # Only CPU torch packages are provided on Mac
    cat neuropods.egg-info/requires.txt | sed '/^\[/ d' | paste -sd " " - | xargs pip install -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
else
    # Install torch compatible with CUDA 10.0
    cat neuropods.egg-info/requires.txt | sed '/^\[/ d' | paste -sd " " - | xargs pip install -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html
fi
popd
