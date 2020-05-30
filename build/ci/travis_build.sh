#!/bin/bash
set -e

# Defaults to python 2 if not set
NEUROPOD_PYTHON_BINARY="python${NEUROPOD_PYTHON_VERSION}"

# Install system dependencies
./build/install_system_deps.sh

# Mac and Travis CI specific deps
if [[ $(uname -s) == 'Darwin' ]]; then
    # Install the requested version of python
    export HOMEBREW_NO_AUTO_UPDATE=1
    brew unlink python

    if [[ "${NEUROPOD_PYTHON_VERSION}" == "2.7" ]]; then
        wget https://www.python.org/ftp/python/2.7.18/python-2.7.18-macosx10.9.pkg &> /dev/null
        sudo installer -pkg python-2.7.18-macosx10.9.pkg -target /
    elif [[ "${NEUROPOD_PYTHON_VERSION}" == "3.5" ]]; then
        wget https://www.python.org/ftp/python/3.5.4/python-3.5.4-macosx10.6.pkg &> /dev/null
        sudo installer -pkg python-3.5.4-macosx10.6.pkg -target /
    elif [[ "${NEUROPOD_PYTHON_VERSION}" == "3.6" ]]; then
        wget https://www.python.org/ftp/python/3.6.8/python-3.6.8-macosx10.9.pkg &> /dev/null
        sudo installer -pkg python-3.6.8-macosx10.9.pkg -target /
    elif [[ "${NEUROPOD_PYTHON_VERSION}" == "3.7" ]]; then
        wget https://www.python.org/ftp/python/3.7.7/python-3.7.7-macosx10.9.pkg &> /dev/null
        sudo installer -pkg python-3.7.7-macosx10.9.pkg -target /
    elif [[ "${NEUROPOD_PYTHON_VERSION}" == "3.8" ]]; then
        wget https://www.python.org/ftp/python/3.8.2/python-3.8.2-macosx10.9.pkg &> /dev/null
        sudo installer -pkg python-3.8.2-macosx10.9.pkg -target /
    fi

    # Install libomp 5
    brew unlink libomp
    brew install https://homebrew.bintray.com/bottles/libomp-5.0.1.high_sierra.bottle.tar.gz
fi

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
