#!/bin/bash
set -e

# Make sure we can build the docs
./build/ci/docs.sh

# Used for bazel caching to s3 in CI
curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g bazels3cache

# Start the S3 build cache
export AWS_ACCESS_KEY_ID=$NEUROPODS_CACHE_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$NEUROPODS_CACHE_ACCESS_SECRET
bazels3cache --bucket=neuropods-build-cache

# Build with the remote cache
./build/build.sh --remote_http_cache=http://localhost:7777

# Install lint deps
./build/ci/install_lint_deps.sh

# Run lint
./build/ci/lint.sh

# Shutdown the cache
curl http://localhost:7777/shutdown

# Install and run fossa
curl -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/fossas/fossa-cli/master/install.sh | bash
pip freeze > source/python/requirements.txt
fossa
