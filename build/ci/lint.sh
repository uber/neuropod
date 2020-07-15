#!/bin/bash
set -e

# Use the virtualenv
source .neuropod_venv/bin/activate

# Generate compile commands
./build/compile_commands.sh

pushd source

# Lint bazel files
python ../build/ci/set_status.py --context "lint/bazel" --description "Bazel Buildifier Lint" \
    /tmp/buildifier --mode check -r .

# Run clang-tidy
# ./bazel-source/external/llvm_toolchain/share/clang/run-clang-tidy.py -clang-tidy-binary ./bazel-source/external/llvm_toolchain/bin/clang-tidy

# Run clang-format
python ../build/ci/set_status.py --context "lint/clang-format" --description "Clang Format Checks" \
    python ../run-clang-format.py --clang-format-executable bazel-source/external/llvm_toolchain/bin/clang-format -r .

pushd python

# Run black
python ../../build/ci/set_status.py --context "lint/black" --description "Black Python Formatting Checks" \
    black --check -t py27 -t py35 -t py36 --diff .

# Run flake8
python ../../build/ci/set_status.py --context "lint/flake8" --description "flake8 Python Style Checks" \
    flake8 .

popd

# Run infer (this is last because it's slow)
python ../build/ci/set_status.py --context "lint/infer" --description "Infer Static Analysis" \
    infer --fail-on-issue --compilation-database compile_commands.json

popd
