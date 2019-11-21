#!/bin/bash
set -e

# This script attempts to autofix lint errors

pushd source

# Reformat with clang-format
find . -name "*.hh" -o -name "*.cc" | xargs -L1 clang-format -style=file -i

pushd python

# Run black
black -t py27 -t py35 -t py36 neuropods

# Run autopep8 to fix errors for flake8
autopep8 -r --in-place neuropods

popd
popd
