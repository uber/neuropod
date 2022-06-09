import sys
from setuptools import setup, find_packages
from setuptools.dist import Distribution

REQUIRED_PACKAGES = ["numpy", "testpath", "future", "six", "pip-tools"]

if sys.version_info[:2] in ((2, 7), (3, 4)):
    # typing is built in to cpython for >= 3.5
    # TODO(vip): This is only used in tests so split into a separate dependency
    REQUIRED_PACKAGES.append("typing")


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(foo):
        return True


setup(
    name="neuropod",
    version="0.3.0rc7",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={
        "": ["neuropod_native.so", "libneuropod.so", "neuropod_multiprocess_worker"]
    },
    distclass=BinaryDistribution,
)
