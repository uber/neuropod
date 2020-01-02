#
# Uber, Inc. (c) 2019
#

import glob
import os
import platform
import subprocess
import sys
import requests

# The `or` pattern below handles empty strings and unset env variables
# Using a default value only handles unset env variables
# TODO(vip): Don't duplicate this between install_frameworks.py and upload_release.py
REQUESTED_TF_VERSION = os.getenv("NEUROPOD_TENSORFLOW_VERSION") or "1.12.0"
REQUESTED_TORCH_VERSION = os.getenv("NEUROPOD_TORCH_VERSION") or "1.1.0"
IS_GPU = (os.getenv("NEUROPOD_IS_GPU") or None) is not None
CUDA_VERSION = os.getenv("NEUROPOD_CUDA_VERSION") or "10.0"
IS_MAC = platform.system() == "Darwin"

GIT_TAG = os.getenv("TRAVIS_TAG", os.getenv("BUILDKITE_TAG"))
PYTHON_VERSION = "{}{}".format(sys.version_info.major, sys.version_info.minor)
GH_UPLOAD_TOKEN = os.getenv("GH_UPLOAD_TOKEN")

def upload():
    release_id = get_release_id(GIT_TAG)
    platform = "libneuropods-{gpustring}-{os}-{tag}".format(
        gpustring="gpu-cuda-{}".format(CUDA_VERSION) if IS_GPU else "cpu",
        os="macos" if IS_MAC else "linux",
        tag=GIT_TAG,
    )

    # Only upload the main library on one build (because we don't need to upload once per backend version)
    if PYTHON_VERSION == "27" and REQUESTED_TF_VERSION == "1.12.0":
        upload_package("source/bazel-bin/neuropods/libneuropods.tar.gz", release_id, "{}.tar.gz".format(platform))

    # Only upload these on the python 2.7 jobs (because we don't need to upload them twice)
    if PYTHON_VERSION == "27":
        upload_package("source/bazel-bin/neuropods/backends/tensorflow/neuropod_tensorflow_backend.tar.gz", release_id, "{}-tensorflow-{}-backend.tar.gz".format(platform, REQUESTED_TF_VERSION))
        upload_package("source/bazel-bin/neuropods/backends/torchscript/neuropod_torchscript_backend.tar.gz", release_id, "{}-torchscript-{}-backend.tar.gz".format(platform, REQUESTED_TORCH_VERSION))

    # Only upload the python backend for one build per platform
    if REQUESTED_TF_VERSION == "1.12.0":
        # Upload the pythonbridge backend
        upload_package("source/bazel-bin/neuropods/backends/python_bridge/neuropod_pythonbridge_backend.tar.gz", release_id, "{}-python-{}-backend.tar.gz".format(platform, PYTHON_VERSION))

    # The python package is the same across platforms so we'll only upload for one platform
    if REQUESTED_TF_VERSION == "1.12.0" and not IS_GPU and not IS_MAC:
        # Upload the wheel
        whl_path = glob.glob('source/python/dist/*.whl')[0]
        fname = os.path.basename(whl_path)
        upload_package(whl_path, release_id, fname, content_type="application/zip")

def get_release_id(tag_name):
    # https://api.github.com/repos/uber/neuropods/releases/tags/{tag_name}
    release_id = requests.get(
        'https://api.github.com/repos/uber/neuropods/releases/tags/{}'.format(tag_name),
        headers={"Authorization": "token {}".format(GH_UPLOAD_TOKEN)},
    ).json()["id"]
    print("Release ID: {}".format(release_id))
    return release_id

def upload_package(local_path, release_id, asset_filename, content_type="application/gzip"):
    # POST https://uploads.github.com/repos/uber/neuropods/releases/{release_id}/assets?name={asset_filename}
    print("Uploading {}...".format(asset_filename))
    with open(local_path, 'rb') as f:
        r = requests.post(
            "https://uploads.github.com/repos/uber/neuropods/releases/{}/assets?name={}".format(release_id, asset_filename),
            headers={
                "Authorization": "token {}".format(GH_UPLOAD_TOKEN),
                "Content-Type": content_type,
            },
            data=f
        )

        if r.status_code != 201:
            print("Error uploading", r.json())
            raise ValueError("Error uploading {}. Got message: {}".format(asset_filename, r.json()["message"]))


if __name__ == '__main__':
    if not GIT_TAG or not GH_UPLOAD_TOKEN:
        # Don't upload if we don't have a tag or token
        pass
    elif os.getenv("TRAVIS_TAG") is not None and not IS_MAC:
        # Don't push releases from linux on Travis
        # (buildkite pushes all the linux releases)
        pass
    else:
        print("Uploading packages...")
        upload()
