#!/bin/bash
set -e

# Builds neuropod and runs tests in a docker container
# Uses a temporary directory to preserve the bazel cache between builds

# Create the cache volume if it doesn't exist
docker volume create neuropod_cache

# Build the image
docker build --target neuropod-base -f build/neuropod.dockerfile -t neuropod .

if [[ "$1" == "-i" ]]; then
    # Run interactively
    docker run --rm -it -v neuropod_cache:/root/.cache neuropod /bin/bash
else
    # Build and test Neuropod
    docker run --rm -v neuropod_cache:/root/.cache neuropod /bin/bash -c "set -e; build/build.sh --config=coverage $@; build/test.sh --config=coverage $@; build/coverage.sh"
fi
