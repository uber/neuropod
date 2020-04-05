#!/bin/bash
set -e

# Use the virtualenv
source .neuropod_venv/bin/activate

BASE_PATH=`pwd`

# Merge all the python coverage reports and print
pushd source/python
coverage combine
coverage report

# Generate an xml report and fix paths
coverage xml
sed -i "s+$BASE_PATH/++g" coverage.xml
sed -i 's+name="\..*neuropod\.python+name="neuropod+g' coverage.xml

popd

# Merge all the coverage reports
source/bazel-source/external/llvm_toolchain/bin/llvm-profdata merge -output=/tmp/neuropod_coverage/code.profdata /tmp/neuropod_coverage/code-*.profraw

# Generate a coverage report
pushd source
bazel query 'kind("cc_binary|cc_test", ...)' | sed 's/\/\//-object bazel-bin\//g' |  sed 's/:/\//g' | paste -sd ' ' | xargs ./bazel-source/external/llvm_toolchain/bin/llvm-cov report -instr-profile=/tmp/neuropod_coverage/code.profdata -ignore-filename-regex="(external|tests)"

# Generate a report to upload and fix paths
bazel query 'kind("cc_binary|cc_test", ...)' | sed 's/\/\//-object bazel-bin\//g' |  sed 's/:/\//g' | paste -sd ' ' | xargs ./bazel-source/external/llvm_toolchain/bin/llvm-cov show -instr-profile=/tmp/neuropod_coverage/code.profdata -ignore-filename-regex="(external|tests)" > coverage.txt
sed -i 's+/proc/self/cwd/+source/+g' coverage.txt
popd
