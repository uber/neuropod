#!/bin/bash
set -e

# Defaults to python 2 if not set
NEUROPOD_PYTHON_BINARY="python${NEUROPOD_PYTHON_VERSION}"


if [[ $(uname -s) == 'Darwin' ]]; then
    export HOMEBREW_NO_AUTO_UPDATE=1

    # Uninstall bazel (because we're going to install bazelisk below)
    brew unlink bazel
fi

# Install system dependencies
./build/install_system_deps.sh

# Mac specific deps
if [[ $(uname -s) == 'Darwin' ]]; then
    # Install the requested version of python
    brew unlink python

    if [[ "${NEUROPOD_PYTHON_VERSION}" == "3.5" ]]; then
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

# Start a bazel cache
wget https://github.com/buchgr/bazel-remote/releases/download/v1.1.0/bazel-remote-1.1.0-darwin-x86_64 -O /tmp/bazel-remote
chmod +x /tmp/bazel-remote
/tmp/bazel-remote --dir /tmp/bazel_cache --max_size 5 --port 7777 &> /dev/null &
CACHE_PID=$!

# Build with the remote cache
./build/build.sh --remote_http_cache=http://localhost:7777

# Run tests
./build/test.sh

# Shutdown the bazel cache
kill $CACHE_PID
