#!/bin/bash
set -e

# Build a docker image that contains a manylinux2010 sysroot
docker build -t neuropod_sysroot .

# Make a directory to put the sysroot in
mkdir -p `pwd`/export

# Copy the sysroot to the export directory
docker run --rm -v `pwd`/export:/export neuropod_sysroot /bin/bash -c "cp /sysroot.tar.gz /export/"