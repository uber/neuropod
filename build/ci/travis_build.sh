#!/bin/bash
set -e

# Default to python 2 if not set
NEUROPOD_PYTHON_BINARY="${NEUROPOD_PYTHON_BINARY:-python}"

# Install system dependencies
./build/install_system_deps.sh

# Mac and Travis CI specific deps
if [[ $(uname -s) == 'Darwin' ]]; then
    # Install python 3.6 (Newest version of py3 that supports TF 1.12.0)
    export HOMEBREW_NO_AUTO_UPDATE=1
    brew unlink python
    wget https://www.python.org/ftp/python/3.6.8/python-3.6.8-macosx10.9.pkg &> /dev/null
    sudo installer -pkg python-3.6.8-macosx10.9.pkg -target /

    # Install libomp 5
    brew unlink libomp
    brew install https://homebrew.bintray.com/bottles/libomp-5.0.1.high_sierra.bottle.tar.gz
fi

# Do everything in a virtualenv
sudo ${NEUROPOD_PYTHON_BINARY} -m pip install virtualenv==16.7.9
${NEUROPOD_PYTHON_BINARY} -m virtualenv /tmp/neuropod_venv
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
export AWS_ACCESS_KEY_ID=$NEUROPOD_CACHE_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$NEUROPOD_CACHE_ACCESS_SECRET
bazels3cache --bucket=neuropod-build-cache

# Build with the remote cache
./build/build.sh --remote_http_cache=http://localhost:7777

# Run tests
./build/test.sh
