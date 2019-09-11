#!/bin/bash
set -e

# The backends need to be on the library path for the tests to dynamically load them
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/bazel-bin/neuropods/backends/python_bridge/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/bazel-bin/neuropods/backends/torchscript/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/bazel-bin/neuropods/backends/tensorflow/

# The worker process for the multiprocess backend needs to be on the path
export PATH=$PATH:`pwd`/bazel-bin/neuropods/multiprocess/

# Ignore ODR errors from ASAN
# See https://github.com/google/sanitizers/wiki/AddressSanitizerOneDefinitionRuleViolation
export ASAN_OPTIONS=detect_odr_violation=0

# Set the ASAN symbolizer path
# See https://clang.llvm.org/docs/AddressSanitizer.html
export ASAN_SYMBOLIZER_PATH=`pwd`/bazel-source/external/llvm_toolchain/bin/llvm-symbolizer

# So the python tests can find the native bindings
export PYTHONPATH=$PYTHONPATH:`pwd`/bazel-bin/neuropods/bindings

# Add the python libray to the pythonpath
export PYTHONPATH=$PYTHONPATH:`pwd`/python

if [[ $(uname -s) == 'Linux' ]]; then
    export TF_CXX=g++-4.8
fi
