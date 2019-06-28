#!/bin/bash
set -e

# Ignore ODR errors from ASAN
# See https://github.com/google/sanitizers/wiki/AddressSanitizerOneDefinitionRuleViolation
export ASAN_OPTIONS=detect_odr_violation=0

# Set the ASAN symbolizer path
# See https://clang.llvm.org/docs/AddressSanitizer.html
export ASAN_SYMBOLIZER_PATH=`pwd`/bazel-source/external/llvm_toolchain/bin/llvm-symbolizer
