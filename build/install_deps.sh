#!/bin/bash
set -e

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    export HOMEBREW_NO_AUTO_UPDATE=1
    brew unlink boost

    brew install boost-python@1.59
    brew link boost@1.59 --force
    brew link boost-python@1.59 --force

    brew install eigen
    brew install jsoncpp
    # brew install libomp
else
    echo "Installing linux deps"
fi
