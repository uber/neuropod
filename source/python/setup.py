from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "numpy",
    "testpath",
    "six"
]

EXTRA_REQUIRE = {
    "tensorflow": ["tensorflow"],
    "torch": ["torch==1.0.0", "torchvision"],
}

setup(
    name="neuropods",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_REQUIRE,
    packages=find_packages(),
)
