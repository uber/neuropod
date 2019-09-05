#!/bin/bash
set -e

# Merge all the python coverage reports and print
pushd source/python
coverage combine
coverage report
popd

# Merge all the coverage reports
source/bazel-source/external/llvm_toolchain/bin/llvm-profdata merge -output=/tmp/neuropod_coverage/code.profdata /tmp/neuropod_coverage/code-*.profraw

# Generate a coverage report
pushd source
bazel query 'kind("cc_binary|cc_test", ...)' | sed 's/\/\//-object bazel-bin\//g' |  sed 's/:/\//g' | paste -sd ' ' | xargs ./bazel-source/external/llvm_toolchain/bin/llvm-cov report -instr-profile=/tmp/neuropod_coverage/code.profdata -ignore-filename-regex="(external|tests)"
popd
