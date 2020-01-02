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
./build/build.sh --remote_http_cache=http://localhost:7777

# Run tests
./build/test.sh

# Run coverage
./build/coverage.sh

# Upload to codecov
bash <(curl -s https://codecov.io/bash) -X coveragepy

# Shutdown the cache
curl http://localhost:7777/shutdown
