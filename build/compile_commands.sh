#!/bin/bash
set -e

echo "Generating compile commands..."

mkdir -p /tmp/bazel-compdb

# Get bazel-compdb if necessary
if [ ! -f "/tmp/bazel-compdb/bazel-compdb" ]; then
    pushd /tmp/bazel-compdb
    # Install bazel-compdb
    VERSION="0.4.2"
    (
        curl -L "https://github.com/grailbio/bazel-compilation-database/archive/${VERSION}.tar.gz" | tar -xz \
        && ln -f -s "/tmp/bazel-compdb/bazel-compilation-database-${VERSION}/generate.sh" bazel-compdb
    )
    popd
fi

pushd source

# Generate compile commands
/tmp/bazel-compdb/bazel-compdb

# Filter compile commands (and update the directory)
jq ".[].directory = \"`pwd`\" | [.[] | select(.file | (contains(\".so\") or contains(\"external/\") or endswith(\".hh\") or endswith(\".h\")) | not)]" compile_commands.json > compile_commands.json2
mv compile_commands.json2 compile_commands.json


popd
