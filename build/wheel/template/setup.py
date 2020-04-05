#
# Uber, Inc. (c) 2020
#

from setuptools import setup, find_packages

REQUIRED_PACKAGES = ["neuropod=={NEUROPOD_VERSION}"]

setup(
    name="{PACKAGE_NAME}",
    version="{NEUROPOD_VERSION}",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={{'': {LIBS}}},
)
