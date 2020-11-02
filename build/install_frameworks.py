# Copyright (c) 2020 UATC, LLC
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

# Installs the appropriate pip packages depending on the following env variables
# NEUROPOD_IS_GPU
# NEUROPOD_TORCH_VERSION
# NEUROPOD_TENSORFLOW_VERSION
import os
import platform
import subprocess
import sys

# The `or` pattern below handles empty strings and unset env variables
# Using a default value only handles unset env variables
REQUESTED_TF_VERSION = os.getenv("NEUROPOD_TENSORFLOW_VERSION") or "1.12.0"
REQUESTED_TORCH_VERSION = os.getenv("NEUROPOD_TORCH_VERSION") or "1.1.0"
IS_GPU = (os.getenv("NEUROPOD_IS_GPU") or None) is not None
CUDA_VERSION = os.getenv("NEUROPOD_CUDA_VERSION") or "10.0"
IS_MAC = platform.system() == "Darwin"

def pip_install(args):
    cmd = [sys.executable, "-m", "pip", "install"] + args
    print("Running pip command: {}".format(cmd))
    subprocess.check_call(cmd)

def install_pytorch(version):
    """
    :param  version:    The version of torch. This can be something like "1.2.0" or
                        "1.1.0.dev20190601"
    """
    pip_args = []

    # Get the torch cuda string (e.g. cpu, cu90, cu92, cu100)
    torch_cuda_string = "cu{}".format(CUDA_VERSION.replace(".", "")) if IS_GPU else "cpu"

    # The base version of torch (e.g. 1.2.0)
    version_base = None

    # If this is a nightly build, what's the date (e.g. 20190809)
    version_date = None

    # Get the version info
    if "dev" in version:
        version_base, version_date = version.split(".dev")
    else:
        version_base = version

    if version_date != None:
        # This is a nightly build
        pip_args += ["-f", "https://download.pytorch.org/whl/nightly/" + torch_cuda_string + "/torch_nightly.html"]
    else:
        # This is a stable build
        pip_args += ["-f", "https://download.pytorch.org/whl/torch_stable.html"]

    # Mac builds do not have the cuda string as part of the version
    if not IS_MAC:
        # If this is the 1.2.0 stable release or it's a nightly build after they started adding the cuda string to the packages
        if (version_base == "1.2.0" and version_date is None) or (version_date != None and int(version_date) > 20190723):
            # For CUDA 10 builds, they don't add `cu100` to the version string
            if torch_cuda_string != "cu100":
                version += "+" + torch_cuda_string

        # If this is the 1.3.0 or 1.4.0 stable release
        if (version_base == "1.3.0" or version_base == "1.4.0") and version_date is None:
            # They changed the default from cuda 10.0 to cuda 10.1
            # For CUDA 10.1 builds, they don't add `cu101` to the version string
            if torch_cuda_string != "cu101":
                version += "+" + torch_cuda_string

        # If this is the 1.5.0, 1.6.0, or 1.7.0 stable release
        if version_base in ["1.5.0", "1.6.0", "1.7.0"] and version_date is None:
            # They changed the default from cuda 10.1 to cuda 10.2
            # For CUDA 10.2 builds, they don't add `cu102` to the version string
            if torch_cuda_string != "cu102":
                version += "+" + torch_cuda_string

    # The Mac 1.3.0 stable release doesn't exist in `torch_stable.html`
    # Use 1.3.0.post2 instead
    if IS_MAC and version_base == "1.3.0" and version_date is None:
        version = "1.3.0.post2"

    if version_date != None:
        if int(version_date) >= 20190802:
            pip_args += ["torch==" + version]
        else:
            pip_args += ["torch_nightly==" + version]
    else:
        if IS_GPU and (version_base == "1.1.0" or version_base == "1.4.0" or version_base == "1.5.0"):
            # See https://github.com/pytorch/pytorch/issues/37113
            # Manually figure out the correct whl URL
            package_version_map = {
                (2,7): "cp27-cp27mu",
                (3,5): "cp35-cp35m",
                (3,6): "cp36-cp36m",
                (3,7): "cp37-cp37m",
                (3,8): "cp38-cp38",
            }
            platform_version = package_version_map[(sys.version_info.major, sys.version_info.minor)]

            pip_args += ["https://download.pytorch.org/whl/" + torch_cuda_string + "/torch-" + version.replace("+", "%2B") + "-" + platform_version + "-linux_x86_64.whl"]
        else:
            pip_args += ["torch==" + version]

    pip_install(pip_args)


def install_tensorflow(version):
    if "dev" in version:
        package = "tf-nightly"
    else:
        package = "tensorflow"

    if IS_GPU:
        package += "-gpu"

    pip_install([package + "==" + version])

if __name__ == '__main__':
    print("Installing tensorflow", REQUESTED_TF_VERSION, "and torch", REQUESTED_TORCH_VERSION)
    install_tensorflow(REQUESTED_TF_VERSION)
    install_pytorch(REQUESTED_TORCH_VERSION)
