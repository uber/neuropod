#!/bin/bash
set -e

# This script runs local lint tools
# Does not run more heavyweight tools like infer or clang-tidy

# Get run-clang-format if necessary
if [ ! -f "/tmp/run-clang-format.py" ]; then
    wget https://raw.githubusercontent.com/Sarcasm/run-clang-format/de6e8ca07d171a7f378d379ff252a00f2905e81d/run-clang-format.py -O /tmp/run-clang-format.py
fi

pushd source

# Run clang-format
python /tmp/run-clang-format.py -r .

pushd python

# Run black in check mode
black --check -t py27 -t py35 -t py36 --diff neuropods

# Run flake8
flake8 neuropods

popd
popd
