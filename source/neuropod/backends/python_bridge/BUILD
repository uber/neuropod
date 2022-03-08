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

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("//bazel:cc.bzl", "neuropod_cc_binary", "neuropod_cc_library")
load("@python_repo//:neuropod_backend_path.bzl", "NEUROPOD_BACKEND_PATH")

neuropod_cc_binary(
    name = "libneuropod_pythonbridge_backend.so",
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [
            "-Wl,-rpath,@loader_path/Python.framework",
        ],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN/usr/lib"],
    }),
    linkshared = True,
    linkstatic = True,
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":python_bridge",
    ],
)

neuropod_cc_library(
    name = "python_bridge",
    srcs = [
        "python_bridge.cc",
        "python_bridge.hh",
    ],
    linkopts = [
        "-ldl",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/backends:neuropod_backend",
        "//neuropod/bindings",
        "//neuropod/core",
        "//neuropod/internal",
        "@filesystem_repo//:filesystem",
        "@python_repo//:python",
    ],
    alwayslink = True,
)

pkg_tar(
    name = "neuropod_pythonbridge_backend",
    srcs = [
        ":libneuropod_pythonbridge_backend.so",
    ],
    extension = "tar.gz",
    package_dir = NEUROPOD_BACKEND_PATH,
    tags = ["manual"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//neuropod/backends/python_bridge/_neuropod_native_bootstrap:neuropod_native_bootstrap",
        "@python_repo//:python.tar.gz",
    ],
)
