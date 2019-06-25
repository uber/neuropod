#!/bin/bash
set -e

# Add TF and torch to the LD_LIBRARY_PATH
# This is so the tests can find the shared objects at runtime
# TODO(vip): find a better way to do this
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/bazel-source/external/libtorch_repo/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/bazel-source/external/tensorflow_repo/lib

# Ignore ODR errors from ASAN
# See https://github.com/google/sanitizers/wiki/AddressSanitizerOneDefinitionRuleViolation
export ASAN_OPTIONS=detect_odr_violation=0

# Set the ASAN symbolizer path
# See https://clang.llvm.org/docs/AddressSanitizer.html
export ASAN_SYMBOLIZER_PATH=`pwd`/bazel-source/external/llvm_toolchain/bin/llvm-symbolizer
