#!/bin/bash
set -e

pushd source

# Generate compile commands
bazel-compdb

# Filter compile commands
jq '[.[] | select(.file | (contains(".so") or contains("external/") or contains(".hh")) | not)]' compile_commands.json > compile_commands.json2
mv compile_commands.json2 compile_commands.json

# Run infer
python ../build/ci/set_status.py --context "lint/infer" --description "Infer Static Analysis" \
    infer --fail-on-issue --compilation-database compile_commands.json

# Run clang-tidy
# ./bazel-source/external/llvm_toolchain/share/clang/run-clang-tidy.py -clang-tidy-binary ./bazel-source/external/llvm_toolchain/bin/clang-tidy

# Run clang-format
python ../build/ci/set_status.py --context "lint/clang-format" --description "Clang Format Checks" \
    python ../run-clang-format.py --clang-format-executable bazel-source/external/llvm_toolchain/bin/clang-format -r .

pushd python

# Run black
python ../../build/ci/set_status.py --context "lint/black" --description "Black Python Formatting Checks" \
    black --check -t py27 -t py35 -t py36 --diff neuropod

# Run flake8
python ../../build/ci/set_status.py --context "lint/flake8" --description "flake8 Python Style Checks" \
    flake8 neuropod

popd
popd
