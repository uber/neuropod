
import os
import platform
import sys
import subprocess
import re
import tarfile

from tempfile import mkdtemp
from testpath.tempdir import TemporaryDirectory

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template")
IS_MAC = platform.system() == "Darwin"

def package(package_name, tar_path, platforms):
    # Remove all `.` characters from the name
    package_name = package_name.replace(".", "-")

    # Create a temporary directory
    tmpdir = mkdtemp()
    print("Packaging in", tmpdir)

    # Make a directory for the package
    package_path = os.path.join(tmpdir, package_name)
    os.mkdir(package_path)

    # Load the tar file containing the backend binaries
    libs = tarfile.open(tar_path)
    members = [item.lstrip("./") for item in libs.getnames() if ".so" in item or ".dylib" in item]

    # Get the backend library
    pattern = re.compile("^libneuropod_.+_backend\.so$")
    shared_library_name = [item for item in members if pattern.match(item)]

    if len(shared_library_name) != 1:
        raise ValueError("The supplied tar file does not contain a neuropod backend")

    shared_library_name = shared_library_name[0]

    print("Packaging backend: ", shared_library_name)
    print("Packaging libraries", members)
    libs.extractall(package_path)

    # Fix the library paths on mac
    if IS_MAC:
        path = os.path.join(package_path, shared_library_name)

        # TODO(vip): Do this a better way
        old_libpath = subprocess.check_output("otool -L {} | grep libneuropod.so | cut -d ' ' -f1 | column -t".format(path), shell=True).strip()
        subprocess.check_call(["chmod", "755", path])
        print("Changing library path from {} for {}".format(old_libpath, path))
        subprocess.check_call(["install_name_tool", "-change", old_libpath, "@executable_path/libneuropod.so", path])

    # For every item in the template directory
    for fname in os.listdir(TEMPLATE_DIR):
        # Get the path to the file
        path = os.path.join(TEMPLATE_DIR, fname)

        # Read the content
        with open(path, 'r') as f:
            content = f.read()

        # Fill in the template
        content = content.format(
            PACKAGE_NAME=package_name,
            # TODO(vip): Update this to get a version from a central location
            NEUROPOD_VERSION="0.1.0",
            SHARED_LIBRARY_NAME=shared_library_name,
            LIBS=str(members),
            PLATFORM=str(platforms),
        )

        # Write it out
        target_path = os.path.join(package_path, fname)
        with open(target_path, 'w') as f:
            f.write(content)

    # Move setup.py
    os.rename(
        os.path.join(package_path, "setup.py"),
        os.path.join(tmpdir, "setup.py")
    )

    # Run it
    subprocess.check_call([sys.executable, "setup.py", "bdist_wheel"], cwd=tmpdir)
    subprocess.check_call([sys.executable, "setup.py", "install"], cwd=tmpdir)


if __name__ == '__main__':
    # The `or` pattern below handles empty strings and unset env variables
    # Using a default value only handles unset env variables
    # TODO(vip): Don't duplicate this between install_frameworks.py and upload_release.py
    REQUESTED_TF_VERSION = os.getenv("NEUROPOD_TENSORFLOW_VERSION") or "1.12.0"
    REQUESTED_TORCH_VERSION = os.getenv("NEUROPOD_TORCH_VERSION") or "1.1.0"
    IS_GPU = (os.getenv("NEUROPOD_IS_GPU") or None) is not None
    CUDA_VERSION = os.getenv("NEUROPOD_CUDA_VERSION") or "10.0"
    PYTHON_VERSION = "{}{}".format(sys.version_info.major, sys.version_info.minor)
    gpustring = "gpu-cuda-{}".format(CUDA_VERSION) if IS_GPU else "cpu"

    # Build wheels for the backends
    package("neuropod-backend-tensorflow-{}-{}".format(REQUESTED_TF_VERSION, gpustring), "bazel-bin/neuropod/backends/tensorflow/neuropod_tensorflow_backend.tar.gz", ["tensorflow"])
    package("neuropod-backend-torchscript-{}-{}".format(REQUESTED_TORCH_VERSION, gpustring), "bazel-bin/neuropod/backends/torchscript/neuropod_torchscript_backend.tar.gz", ["torchscript"])

    # The python backend isn't different depending on CPU/GPU so we don't include that in the name
    package("neuropod-backend-python-{}".format(PYTHON_VERSION), "bazel-bin/neuropod/backends/python_bridge/neuropod_pythonbridge_backend.tar.gz", ["python", "pytorch"])
