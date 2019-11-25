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

# Install node and npm if we need to
if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Used for bazel caching to s3 in CI
sudo npm install -g bazels3cache

# Start the S3 build cache
export AWS_ACCESS_KEY_ID=$NEUROPODS_CACHE_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$NEUROPODS_CACHE_ACCESS_SECRET
bazels3cache --bucket=neuropods-build-cache

# Build with the remote cache
./build/build.sh --remote_http_cache=http://localhost:7777

# Run tests
./build/test.sh
