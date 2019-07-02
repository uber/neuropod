from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "numpy<1.17.0",
    "testpath",
    "future",
    "six"
]

EXTRA_REQUIRE = {
    "tensorflow": ["tensorflow==1.12.0"],
    "torch": ["torch==1.1.0"],
}

setup(
    name="neuropods",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_REQUIRE,
    packages=find_packages(),
)
