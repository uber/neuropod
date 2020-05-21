#
# Uber, Inc. (c) 2020
#

import errno
import os
import glob
import subprocess
import sys

PACKAGE_BASE_DIR = os.path.abspath(
    os.path.expanduser(
        "~/.neuropod/pythonpackages/py{}{}/".format(
            sys.version_info.major, sys.version_info.minor
        )
    )
)


def create_if_not_exists(path):
    # Messy because this needs to be python2 + 3 compatible
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def compile_requirements(requirements, lockfile):
    """
    Run piptools compile over a requirement file to generate an output lockfile.
    We use `--allow-unsafe` as we want to include wheel, pip, and setuptools in our output.
    """
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "piptools",
            "compile",
            "--no-header",
            "--allow-unsafe",
            "-q",
            "-o",
            lockfile,
            requirements,
        ],
    )


def install_package(specifier, prefix):
    """
    Install a pip package (without dependencies) into the prefix directory.
    """

    # TODO(vip): If this command is run as a different user, we can prevent models
    # from modifying the installed packages (can also do things like read-only mounts, etc.)
    from pip._internal.cli.main import main as pip_entry_point

    exit_code = pip_entry_point(
        [
            "-q",
            "install",
            "--compile",
            "--no-deps",
            "--disable-pip-version-check",
            "--no-warn-script-location",
            "--prefix={}".format(prefix),
            "--ignore-installed",
            specifier,
        ]
    )

    if exit_code != 0:
        raise RuntimeError("Error installing python dependency: {}".format(specifier))


def load_deps(lockfile):
    """
    For each dependency in the lockfile, install it to the cachedir if necessary
    and add it to sys.path

    Note: the lockfile contains all transitive deps so we don't need to do any
    recursive scanning

    This is intented to be used by the native code when running with OPE (i.e. one
    model per process).
    """

    requirements = []
    with open(lockfile, "r") as f:
        for line in f:
            # Remove comments
            pos = line.find("#")
            if pos != -1:
                line = line[:pos]

            # Remove surrounding whitespace
            line = line.strip()
            if not line:
                continue

            # Anything else must be of the form name==version or we error
            parts = line.split("==")
            if len(parts) != 2:
                raise ValueError(
                    "Expected requirements of the form name==version but got {}".format(
                        line
                    )
                )

            # Add it to our list of requirements
            requirements.append(line.lower())

    # Create our package base dir if necessary
    create_if_not_exists(PACKAGE_BASE_DIR)

    for requirement in requirements:
        req_path = os.path.abspath(os.path.join(PACKAGE_BASE_DIR, requirement))

        # Sanity check to guard against bad input
        if not req_path.startswith(PACKAGE_BASE_DIR):
            raise ValueError("Invalid dependency: {}".format(requirement))

        # Install this package if we need to
        if not os.path.isdir(req_path):
            install_package(requirement, req_path)

        # Add this package to the pythonpath
        sys.path.insert(
            0, glob.glob("{}/lib/python*/site-packages".format(req_path))[0]
        )
