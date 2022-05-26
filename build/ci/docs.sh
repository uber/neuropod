#!/bin/bash
set -e

# Use the virtualenv
source .neuropod_venv/bin/activate

# Install deps for the docs
pip install mkdocs-material=="8.*"

# Make sure we can build the docs
python ./build/ci/set_status.py --context "docs/build" --description "Build the docs" \
    ./build/docs.sh

# Deploy docs if we're running on the master branch
if [[ "$BUILDKITE_BRANCH" == "master" ]]; then

# Get the repo root
REPO_ROOT=`pwd`

# Get the path of the docs
pushd build/docs/_static
DOCS_DIR=`pwd`
popd

# Switch to a tempdir
tmpdir=$(mktemp -d)
pushd $tmpdir

# Setup keys used for deployment (only works on the master branch)
eval "$(ssh-agent -s)"
ssh-add - <<< "$WEB_DEPLOY_KEY"
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Checkout the docs
git clone --branch docs --single-branch git@github.com:uber/neuropod.git
cd neuropod

# Make sure we're not deploying an older version of the docs
LAST_REVISION="$(cat master_revision)"
pushd "$REPO_ROOT"

# Returns nonzero exit code if it's not an ancestor and the `set -e` at the top
# will cause this to abort
git merge-base --is-ancestor "$LAST_REVISION" "$BUILDKITE_COMMIT"
popd

# Write the current revision
echo "$BUILDKITE_COMMIT" > master_revision

# Remove the old docs
pushd site/docs
rm -rf master

# Copy in the new ones
mv "$DOCS_DIR" master
popd

# Add everything
git add .
git config --global user.email "vip+neuropodbot@uber.com"
git config --global user.name "NeuropodBot"
git commit -m "Build docs from uber/neuropod@$BUILDKITE_COMMIT"

git push origin docs

popd

# Remove tempdir
rm -rf "$tmpdir"

fi
