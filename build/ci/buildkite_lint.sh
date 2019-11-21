#!/bin/bash
set -e

# Install lint deps
./build/ci/install_lint_deps.sh

# Run lint
./build/ci/lint.sh
