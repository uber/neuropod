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

cc_library(
    name = "serialization",
    hdrs = [
        "serialization.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "@boost_repo//:boost",
    ],
)

cc_library(
    name = "impl",
    srcs = [
        "serialization.cc",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":serialization",
        "//neuropod/backends:neuropod_backend",
    ],
)

# Package all the header files
pkg_tar(
    name = "libneuropod_serialization_hdrs",
    srcs = glob(["*.hh"]),
    package_dir = "serialization/",
    visibility = [
        "//visibility:public",
    ],
)
