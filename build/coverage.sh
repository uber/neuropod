#!/bin/bash
set -e

# Use the virtualenv
source .neuropod_venv/bin/activate

# So the `coverage` commands find the config
export COVERAGE_RCFILE="`pwd`/source/python/.coveragerc"

# Merge all the python coverage reports
pushd /tmp/neuropod_py_coverage/
coverage combine
popd

# Print the report
coverage report

# Generate an xml report and fix paths
coverage xml
NEUROPOD_PY_BOOTSTRAP_DIR=`echo .neuropod_test_base/*/backends/python_*/bootstrap/_neuropod_native_bootstrap/`
sed -i "s+$NEUROPOD_PY_BOOTSTRAP_DIR+source/neuropod/backends/python_bridge/_neuropod_native_bootstrap/+g" coverage.xml

pushd source

# Merge all the coverage reports for native code
./external/llvm_toolchain/bin/llvm-profdata merge -output=/tmp/neuropod_coverage/code.profdata /tmp/neuropod_coverage/code-*.profraw

# Generate a coverage report and fix paths
bazel query 'kind("cc_binary|cc_test", ...)' | sed 's/\/\//-object bazel-bin\//g' |  sed 's/:/\//g' | paste -sd ' ' | xargs ./bazel-source/external/llvm_toolchain/bin/llvm-cov export --format=lcov -instr-profile=/tmp/neuropod_coverage/code.profdata -ignore-filename-regex="(external|test_|benchmark_)" > native_coverage.txt
sed -i 's+/proc/self/cwd/+source/+g' native_coverage.txt

# Get Java coverage and fix paths
cp -f ./bazel-source/bazel-out/_coverage/_coverage_report.dat java_coverage.txt
sed -i 's+SF:neuropod/bindings+SF:source/neuropod/bindings+g' java_coverage.txt

# Combine into one report
lcov -a java_coverage.txt -a native_coverage.txt -o coverage.txt
rm java_coverage.txt native_coverage.txt

# Display coverage information
lcov -l coverage.txt
popd
