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
load("@tensorflow_repo//:neuropod_backend_path.bzl", "NEUROPOD_BACKEND_PATH")

neuropod_cc_binary(
    name = "libneuropod_tensorflow_backend.so",
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
        ":tensorflow_backend",
    ],
)

neuropod_cc_library(
    name = "tensorflow_backend",
    srcs = glob([
        "*.cc",
        "*.hh",
        "saved_model/*.cc",
        "saved_model/*.h",
    ]),
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal",
        "@libjsoncpp_repo//:libjsoncpp",
        "@tensorflow_hdrs_repo//:tensorflow_hdrs",
        "@tensorflow_repo//:libtensorflow",
    ],
    alwayslink = True,
)

pkg_tar(
    name = "neuropod_tensorflow_backend",
    srcs = [
        ":libneuropod_tensorflow_backend.so",
        "@tensorflow_repo//:libtensorflow_libs",
    ],
    extension = "tar.gz",
    package_dir = NEUROPOD_BACKEND_PATH,
    tags = ["manual"],
    visibility = [
        "//visibility:public",
    ],
)
