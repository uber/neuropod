# Copyright (c) 2020 The Neuropod Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import errno
import os
import glob
import sys

from _neuropod_native_bootstrap.filelock import FileLock

PACKAGE_BASE_DIR = os.path.abspath(
    os.path.expanduser(
        "~/.neuropod/pythonpackages/py{}{}/".format(
            sys.version_info.major, sys.version_info.minor
        )
    )
)

LOG_PY_DEPS = os.getenv("NEUROPOD_LOG_PY_DEPS") is not None

def create_if_not_exists(path):
    # Messy because this needs to be python2 + 3 compatible
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


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


def bootstrap_requirements():
    """
    If we're loading the python library from native code in an isolated environment,
    we need to make sure that our required deps are available before we try to load
    Neuropod

    This is called from the PythonBridge in native code
    """

    if hasattr(bootstrap_requirements, "did_run"):
        # Only need to run this once
        return

    bootstrap_requirements.did_run = True

    # A lockfile of runtime requirements to bootstrap with
    reqs = """
    future==0.18.2
    numpy=={}
    six==1.15.0
    testpath==0.4.4
    """.format(
        "1.18.0" if sys.version_info.major == 3 else "1.16.6"
    )

    _load_deps_internal(reqs)


def load_deps(lockfile):
    """
    For each dependency in the lockfile, install it to the cachedir if necessary
    and add it to sys.path

    Note: the lockfile contains all transitive deps so we don't need to do any
    recursive scanning

    This is intented to be used by the native code when running with OPE (i.e. one
    model per process).
    """
    with open(lockfile, "r") as f:
        lockfile_contents = f.read()

    # Load the data
    _load_deps_internal(lockfile_contents)


def _load_deps_internal(lockfile_contents):
    """
    See `load_deps` above for details
    """

    print(
        "Neuropod Info: Installing python packages required by the current model. "
        "If this takes a long time, set NEUROPOD_LOG_PY_DEPS to log all dependencies of the model "
        "and consider preinstalling large dependencies using `neuropod.backends.python.utils.preinstall_deps`"
    )

    requirements = []
    for line in lockfile_contents.splitlines():
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

        # Check if we need to install the package
        if not os.path.exists(req_path + ".complete"):
            if LOG_PY_DEPS:
                print("[Neuropod] Installing python package required by model: {}...".format(requirement))

            # Acquire a lock
            # TODO(vip): it's possible that this'll conflict with a package if the version ends in `.lock`
            lock = FileLock(req_path + ".lock")
            with lock:
                # Check again if we need to install the package
                # (maybe another process installed it before we got the lock)
                if not os.path.exists(req_path + ".complete"):
                    # Install
                    install_package(requirement, req_path)

                    # Mark the installation as complete
                    open(req_path + ".complete", "a").close()
        else:
            if LOG_PY_DEPS:
                print("[Neuropod] Python package required by model already installed: {}...".format(requirement))

        # Add this package to the pythonpath
        sys.path.insert(
            0, glob.glob("{}/lib/python*/site-packages".format(req_path))[0]
        )
