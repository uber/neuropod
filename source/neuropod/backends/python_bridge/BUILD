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

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_binary(
    name = "libneuropod_pythonbridge_backend.so",
    srcs = [
        "//neuropod:libneuropod.so",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
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

cc_library(
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
    tags = ["manual"],
    visibility = [
        "//visibility:public",
    ],
)
