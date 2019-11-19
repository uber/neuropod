#!/bin/bash
set -e

# Run tests
./build/test.sh

# Run coverage
./build/coverage.sh

# Upload to codecov
bash <(curl -s https://codecov.io/bash) -X coveragepy
