#!/bin/bash
set -e

# Use the virtualenv
source .neuropod_venv/bin/activate

# Make sure we can build the docs
python ./build/ci/set_status.py --context "docs/build" --description "Build the docs" \
    ./build/docs.sh

# TODO(vip): After open-sourcing, deploy the docs here
