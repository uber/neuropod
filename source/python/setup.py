from setuptools import setup, find_packages

REQUIRED_PACKAGES = ["numpy<1.17.0", "testpath", "typing", "future", "six"]

setup(
    name="neuropod",
    version="0.1.0",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={'': ["neuropod_native.so", "libneuropod.so", "neuropod_multiprocess_worker"]},
)
