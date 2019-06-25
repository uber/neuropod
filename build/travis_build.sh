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

# In order to avoid wasting resources, all travis builds are built with AddressSanitizer
# (instead of having builds with and without)
# Once we begin releasing artifacts, we can add a release build without ASAN

# Build
./build/build.sh --config=asan

# Run tests
./build/test.sh --config=asan
