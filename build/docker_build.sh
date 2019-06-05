#!/bin/bash
set -e

# Builds neuropods and runs tests in a docker container
# Uses a temporary directory to preserve the bazel cache between builds

# Create the cache folder if it doesn't exist
mkdir -p /tmp/neuropod_docker_cache

# Build the image
docker build -f build/neuropods.dockerfile -t neuropods .

if [[ "$1" == "-i" ]]; then
    # Run interactively
    docker run --rm -it -v /tmp/neuropod_docker_cache:/root/.cache neuropods /bin/bash
else
    # Build and test Neuropods
    docker run --rm -v /tmp/neuropod_docker_cache:/root/.cache neuropods build/build.sh
fi
