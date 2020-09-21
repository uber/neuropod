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
if [[ -z "${BUILDKITE_TAG}" ]]; then
    # No tag so build with coverage
    ./build/build.sh --config=coverage --config=ci --remote_http_cache=http://localhost:7777

    # Run tests
    ./build/test_gpu.sh --config=coverage --config=ci --remote_http_cache=http://localhost:7777
else
    # This is a release
    ./build/build.sh --config=ci --remote_http_cache=http://localhost:7777

    # Run tests
    ./build/test_gpu.sh --config=ci --remote_http_cache=http://localhost:7777
fi

if [[ -z "${BUILDKITE_TAG}" ]]; then
    # Run coverage
    ./build/coverage.sh

    # Upload to codecov
    bash <(curl -s https://codecov.io/bash) -X coveragepy
fi

# Shutdown the cache
curl http://localhost:7777/shutdown
