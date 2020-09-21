#!/bin/bash
set -e

# Used for bazel caching to s3 in CI
curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g bazels3cache

# Start the S3 build cache
export AWS_ACCESS_KEY_ID=$NEUROPOD_CACHE_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$NEUROPOD_CACHE_ACCESS_SECRET
bazels3cache --bucket=neuropod-build-cache

# Build with the remote cache
./build/build.sh --config=ci --remote_http_cache=http://localhost:7777

# Install lint deps
./build/ci/install_lint_deps.sh

# Make sure we can build the docs
./build/ci/docs.sh

# Run lint
./build/ci/lint.sh

# Shutdown the cache
curl http://localhost:7777/shutdown
