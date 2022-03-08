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

GIT_TAG = os.getenv("BUILDKITE_TAG")
if GIT_TAG is None:
    # Check GH actions
    github_ref = os.getenv("GITHUB_REF", default="")
    if github_ref.startswith("refs/tags/"):
        GIT_TAG = github_ref.replace("refs/tags/","", 1)

PYTHON_VERSION = "{}{}".format(sys.version_info.major, sys.version_info.minor)
GH_UPLOAD_TOKEN = os.getenv("GH_UPLOAD_TOKEN")

# Get the frameworks we should upload
NEUROPOD_TEST_FRAMEWORKS = os.getenv("NEUROPOD_TEST_FRAMEWORKS")
if NEUROPOD_TEST_FRAMEWORKS is not None:
    NEUROPOD_TEST_FRAMEWORKS = set(NEUROPOD_TEST_FRAMEWORKS.split(","))

def should_upload(framework):
    """
    Whether we should upload an artifact for `framework`
    """
    if NEUROPOD_TEST_FRAMEWORKS is None:
        return True

    return framework in NEUROPOD_TEST_FRAMEWORKS


def upload():
    release_id = get_release_id(GIT_TAG)
    platform = "libneuropod-{gpustring}-{os}-{tag}".format(
        gpustring="gpu-cuda-{}".format(CUDA_VERSION) if IS_GPU else "cpu",
        os="macos" if IS_MAC else "linux",
        tag=GIT_TAG,
    )

    # Only upload the main library on one build (because we don't need to upload once per backend version)
    # This is also not CPU/GPU dependent
    # For each OS:
    if REQUESTED_TF_VERSION == "1.14.0" and not IS_GPU:
        upload_package(
            "source/bazel-bin/neuropod/libneuropod.tar.gz",
            release_id,
            "libneuropod-{os}-{tag}.tar.gz".format(
                os="macos" if IS_MAC else "linux",
                tag=GIT_TAG,
            )
        )

    if should_upload("torchscript"):
        # For each OS: For each backend version: For each CPU/GPU:
        upload_package("source/bazel-bin/neuropod/backends/torchscript/neuropod_torchscript_backend.tar.gz", release_id, "{}-torchscript-{}-backend.tar.gz".format(platform, REQUESTED_TORCH_VERSION))

    if should_upload("tensorflow"):
        # For each OS: For each backend version: For each CPU/GPU:
        upload_package("source/bazel-bin/neuropod/backends/tensorflow/neuropod_tensorflow_backend.tar.gz", release_id, "{}-tensorflow-{}-backend.tar.gz".format(platform, REQUESTED_TF_VERSION))

    # The python package is the same across CPU/GPU and different versions of backends so we'll only upload once for mac and once for linux
    # TODO(vip): Do this better
    if REQUESTED_TORCH_VERSION not in ["1.6.0", "1.7.0", "1.8.1", "1.9.0"] and not IS_GPU:
        # For each OS: For each python version
        # Upload the pythonbridge backend
        upload_package("source/bazel-bin/neuropod/backends/python_bridge/neuropod_pythonbridge_backend.tar.gz", release_id, "{}-python-{}-backend.tar.gz".format(platform, PYTHON_VERSION))

        # Upload the wheels
        for gpath in ["source/python/dist/neuropod-*.whl"]:
            whl_path = glob.glob(gpath)[0]
            fname = os.path.basename(whl_path)
            upload_package(whl_path, release_id, fname, content_type="application/zip")

def get_release_id(tag_name):
    # https://api.github.com/repos/uber/neuropod/releases/tags/{tag_name}
    release_id = requests.get(
        'https://api.github.com/repos/uber/neuropod/releases/tags/{}'.format(tag_name),
        headers={"Authorization": "token {}".format(GH_UPLOAD_TOKEN)},
    ).json()["id"]
    print("Release ID: {}".format(release_id))
    return release_id

def get_repo_info():
    # https://api.github.com/repos/uber/neuropod
    return requests.get('https://api.github.com/repos/uber/neuropod').json()

def upload_package(local_path, release_id, asset_filename, content_type="application/gzip"):
    # POST https://uploads.github.com/repos/uber/neuropod/releases/{release_id}/assets?name={asset_filename}
    print("Uploading {}...".format(asset_filename))
    with open(local_path, 'rb') as f:
        r = requests.post(
            "https://uploads.github.com/repos/uber/neuropod/releases/{}/assets?name={}".format(release_id, asset_filename),
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
    if os.getenv("CI") is not None:
        # Make a request to the github API to ensure we can (this helps catch failures like old SSL libs)
        get_repo_info()

    if not GIT_TAG or not GH_UPLOAD_TOKEN:
        # Don't upload if we don't have a tag or token
        pass
    else:
        print("Uploading packages for tag {}...".format(GIT_TAG))
        upload()
