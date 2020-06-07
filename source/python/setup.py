from setuptools import setup, find_packages
from setuptools.dist import Distribution

REQUIRED_PACKAGES = ["numpy", "testpath", "typing", "future", "six"]

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True

setup(
    name="neuropod",
    version="0.2.0",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={'': ["neuropod_native.so", "libneuropod.so", "neuropod_multiprocess_worker"]},
    distclass=BinaryDistribution,
)
