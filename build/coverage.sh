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

# Merge all the coverage reports for native code
source/bazel-source/external/llvm_toolchain/bin/llvm-profdata merge -output=/tmp/neuropod_coverage/code.profdata /tmp/neuropod_coverage/code-*.profraw

# Generate a coverage report
pushd source
bazel query 'kind("cc_binary|cc_test", ...)' | sed 's/\/\//-object bazel-bin\//g' |  sed 's/:/\//g' | paste -sd ' ' | xargs ./bazel-source/external/llvm_toolchain/bin/llvm-cov report -instr-profile=/tmp/neuropod_coverage/code.profdata -ignore-filename-regex="(external|tests)"

# Generate a report to upload and fix paths
bazel query 'kind("cc_binary|cc_test", ...)' | sed 's/\/\//-object bazel-bin\//g' |  sed 's/:/\//g' | paste -sd ' ' | xargs ./bazel-source/external/llvm_toolchain/bin/llvm-cov show -instr-profile=/tmp/neuropod_coverage/code.profdata -ignore-filename-regex="(external|tests)" > coverage.txt
sed -i 's+/proc/self/cwd/+source/+g' coverage.txt
popd

pushd source
# Set PATH for Java tests
PATH=$PATH:`pwd`/bazel-bin/neuropod/multiprocess/
# Generate a coverage report for Java
bazel coverage --nocache_test_results --collect_code_coverage --instrumentation_filter='/java[/:]' --combined_report=lcov --coverage_report_generator=@bazel_tools//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Main //neuropod/bindings/java/...

# bazel coverage configured to create a combined LCOV coverage report for java code.
# codecov supports Lcov TXT and so .txt file. Copy and rename it accordingly.
[[ -f ./bazel-source/bazel-out/_coverage/_coverage_report.dat ]] && cp -f ./bazel-source/bazel-out/_coverage/_coverage_report.dat ./_coverage_report.txt
popd
