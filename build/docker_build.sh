#!/bin/bash
set -e

# Builds neuropods and runs tests in a docker container
# Uses a temporary directory to preserve the bazel cache between builds

# Create the cache volume if it doesn't exist
docker volume create neuropod_cache

# Build the image
docker build --target neuropod-base -f build/neuropods.dockerfile -t neuropods .

if [[ "$1" == "-i" ]]; then
    # Run interactively
    docker run --rm -it -v neuropod_cache:/root/.cache neuropods /bin/bash
else
    # Build and test Neuropods
    docker run --rm -v neuropod_cache:/root/.cache neuropods /bin/bash -c "set -e; build/build.sh $@; build/test.sh $@; build/coverage.sh $@"
fi
