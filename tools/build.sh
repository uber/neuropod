#!/bin/bash

# Make a directory for our stable version
# Netlify doesn't support symlinks so we need to make a copy
pushd site/docs
cp -r 0.2.0 stable
popd
