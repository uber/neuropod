#!/bin/bash
set -e

# Generate the python api markdown
pushd source/python
PYTHONPATH=. python ../../build/gen_py_api_docs.py ../../docs/packagers
popd

# Build the docs
mkdocs build -d build/docs/_static
