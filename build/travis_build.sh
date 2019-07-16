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

# Make sure that the CI matrix is correct
# This command will fail if the matrix defined in the script
# does not match the config files
python ./build/ci_matrix.py

# Build
./build/build.sh

# Run tests
./build/test.sh
