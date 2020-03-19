#!/bin/bash
set -e

# This script runs clang-tidy locally

# Generate compile commands
# TODO(vip): Don't do this every time
./build/compile_commands.sh

pushd source

# Run clang-tidy
# TODO(vip): We can run in parallel with the below command, but it doesn't show the output in color
# ./external/llvm_toolchain/share/clang/run-clang-tidy.py -j 16 -clang-tidy-binary ./external/llvm_toolchain/bin/clang-tidy
jq '.[].file' compile_commands.json | xargs -L 1 ./external/llvm_toolchain/bin/clang-tidy


popd
