from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "numpy<1.17.0",
    "testpath",
    "future",
    "six"
]

setup(
    name="neuropods",
    version="0.1.0",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
)
