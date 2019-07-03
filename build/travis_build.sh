#!/bin/bash
set -e

# Do everything in a virtualenv
sudo pip install -U pip
sudo pip install virtualenv
virtualenv /tmp/neuropod_venv
source /tmp/neuropod_venv/bin/activate

# Install dependencies
./build/install_system_deps.sh
./build/install_python_deps.sh

# Build
./build/build.sh

# Run tests
./build/test.sh

# Upload code coverage reports
pushd source
bazel coverage --cache_test_results=no --test_output=errors "$@" //...
popd

bash <(curl -s https://codecov.io/bash) -s source/bazel-testlogs/
