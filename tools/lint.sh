#!/bin/bash
set -e

# This script runs local lint tools
# Does not run more heavyweight tools like infer or clang-tidy

# Get run-clang-format if necessary
if [ ! -f "/tmp/run-clang-format.py" ]; then
    wget https://raw.githubusercontent.com/Sarcasm/run-clang-format/de6e8ca07d171a7f378d379ff252a00f2905e81d/run-clang-format.py -O /tmp/run-clang-format.py
fi

# Check bazel files
if [[ $(uname -s) == 'Darwin' ]]; then
    if [ ! -f "/tmp/buildifier.mac" ]; then
        wget https://github.com/bazelbuild/buildtools/releases/download/2.2.1/buildifier.mac -O /tmp/buildifier.mac
        chmod +x /tmp/buildifier.mac
    fi

    /tmp/buildifier.mac --mode diff -r source
else
    if [ ! -f "/tmp/buildifier" ]; then
        wget https://github.com/bazelbuild/buildtools/releases/download/2.2.1/buildifier -O /tmp/buildifier
        chmod +x /tmp/buildifier
    fi

    /tmp/buildifier --mode diff -r source
fi

pushd source

# Run clang-format
# Note: this assumes the script is being run from the root of the repo
python /tmp/run-clang-format.py --clang-format-executable bazel-source/external/llvm_toolchain/bin/clang-format -r .

pushd python

# Run black in check mode
black --check -t py27 -t py35 -t py36 --diff .

# Run flake8
flake8 .

popd
popd
