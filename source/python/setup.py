from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "numpy",
    "testpath",
    "future",
    "six"
]

EXTRA_REQUIRE = {
    "tensorflow": ["tensorflow"],
    "torch": ["torch_nightly==1.0.0.dev20190318"],
}

setup(
    name="neuropods",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_REQUIRE,
    packages=find_packages(),
)
