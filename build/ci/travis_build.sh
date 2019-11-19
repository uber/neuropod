#!/bin/bash
set -e

# Default to python 2 if not set
NEUROPODS_PYTHON_BINARY="${NEUROPODS_PYTHON_BINARY:-python}"

# Install system dependencies
./build/install_system_deps.sh

# Do everything in a virtualenv
sudo ${NEUROPODS_PYTHON_BINARY} -m pip install virtualenv
${NEUROPODS_PYTHON_BINARY} -m virtualenv /tmp/neuropod_venv
source /tmp/neuropod_venv/bin/activate

# Install python dependencies
./build/install_python_deps.sh

# Make sure that the CI matrix is correct
# This command will fail if the matrix defined in the script
# does not match the config files
python ./build/ci_matrix.py

# Build
./build/build.sh

# Run tests
./build/test.sh
