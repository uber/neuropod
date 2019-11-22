#!/bin/bash
set -e

# Make sure we can build the docs
./build/ci/docs.sh

# Install lint deps
./build/ci/install_lint_deps.sh

# Run lint
./build/ci/lint.sh
