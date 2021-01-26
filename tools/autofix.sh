#!/bin/bash
set -e

# This script attempts to autofix lint errors

# Try to fix bazel files
# Note: this assumes that `lint.sh` has been run to download the deps
if [[ $(uname -s) == 'Darwin' ]]; then
    /tmp/buildifier.mac -r source
else
    /tmp/buildifier -r source
fi

pushd source

# Reformat with clang-format
# Note: this assumes the script is being run from the root of the repo
find . -name "*.hh" -o -name "*.h" -o -name "*.cc" -o -name "*.c" | xargs -L1 bazel-source/external/llvm_toolchain/bin/clang-format -style=file -i

pushd python

# Run black
black -t py27 -t py35 -t py36 .

# Run autopep8 to fix errors for flake8
autopep8 -r --in-place .

popd
popd
