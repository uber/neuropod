#
# Uber, Inc. (c) 2019
#

# Installs the appropriate pip packages depending on the following env variables
# NEUROPODS_IS_GPU
# NEUROPODS_TORCH_VERSION
# NEUROPODS_TENSORFLOW_VERSION
import os
import platform
import subprocess
import sys

# The `or` pattern below handles empty strings and unset env variables
# Using a default value only handles unset env variables
REQUESTED_TF_VERSION = os.getenv("NEUROPODS_TENSORFLOW_VERSION") or "1.12.0"
REQUESTED_TORCH_VERSION = os.getenv("NEUROPODS_TORCH_VERSION") or "1.1.0"
IS_GPU = (os.getenv("NEUROPODS_IS_GPU") or None) is not None
CUDA_VERSION = os.getenv("NEUROPODS_CUDA_VERSION") or "10.0"
IS_MAC = platform.system() == "Darwin"

def pip_install(args):
    cmd = [sys.executable, "-m", "pip", "install"] + args
    print "Running pip command: {}".format(cmd)
    subprocess.check_call(cmd)

def install_pytorch(version):
    pip_args = []
    torch_cuda_string = "cu{}".format(CUDA_VERSION.replace(".", ""))

    if "dev" in version:
        if IS_GPU:
            pip_args += ["-f", "https://download.pytorch.org/whl/nightly/" + torch_cuda_string + "/torch_nightly.html"]
        else:
            pip_args += ["-f", "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"]

        pip_args += ["torch_nightly==" + version]
    else:
        if IS_GPU:
            pip_args += ["https://download.pytorch.org/whl/" + torch_cuda_string + "/torch-" + version + "-cp27-cp27mu-linux_x86_64.whl"]
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
    print "Installing tensorflow", REQUESTED_TF_VERSION, "and torch", REQUESTED_TORCH_VERSION
    install_tensorflow(REQUESTED_TF_VERSION)
    install_pytorch(REQUESTED_TORCH_VERSION)
