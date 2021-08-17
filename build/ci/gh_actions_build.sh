#!/bin/bash
set -e

# Defaults to python 2 if not set
NEUROPOD_PYTHON_BINARY="python${NEUROPOD_PYTHON_VERSION}"

# Install system dependencies
./build/install_system_deps.sh

# Mac specific deps
if [[ $(uname -s) == 'Darwin' ]]; then
    # Install the requested version of python
    export HOMEBREW_NO_AUTO_UPDATE=1
    brew unlink python

    if [[ "${NEUROPOD_PYTHON_VERSION}" == "2.7" ]]; then
        wget https://www.python.org/ftp/python/2.7.18/python-2.7.18-macosx10.9.pkg &> /dev/null
        sudo installer -pkg python-2.7.18-macosx10.9.pkg -target /
    elif [[ "${NEUROPOD_PYTHON_VERSION}" == "3.5" ]]; then
        # SSL is broken on the official python 3.5 release (and python 3.5 is deprecated)
        # so we need to install it a different way
        export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-10.15}"
        brew install pyenv
        pyenv install 3.5.4
        pyenv global 3.5.4
        eval "$(pyenv init --path)"
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

# Get and start a bazel cache
if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    wget https://github.com/buchgr/bazel-remote/releases/download/v1.2.0/bazel-remote-1.2.0-linux-x86_64 -O /tmp/bazel-remote
else
    wget https://github.com/buchgr/bazel-remote/releases/download/v1.2.0/bazel-remote-1.2.0-darwin-x86_64 -O /tmp/bazel-remote
fi

chmod +x /tmp/bazel-remote
/tmp/bazel-remote --dir ~/bazel_cache --max_size 5 --port 7777 &> /dev/null &
CACHE_PID=$!

# Build with the remote cache
./build/build.sh --config=ci --remote_http_cache=http://localhost:7777

# Run tests
./build/test.sh --config=ci --remote_http_cache=http://localhost:7777

# Shutdown the bazel cache
kill $CACHE_PID
