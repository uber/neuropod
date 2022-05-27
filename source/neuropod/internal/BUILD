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
load("//bazel:cc.bzl", "neuropod_cc_library")

neuropod_cc_library(
    name = "internal",
    hdrs = glob(["*.hh"]),
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod:options",
        "@fmt_repo//:fmt",
        "@spdlog_repo//:spdlog",
    ],
)

neuropod_cc_library(
    name = "impl",
    srcs = glob(["*.cc"]),
    linkopts = ["-ldl"],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":internal",
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends:neuropod_backend",
        "//neuropod/serialization",
        "@boost_repo//:boost",
        "@filesystem_repo//:filesystem",
        "@libjsoncpp_repo//:libjsoncpp",
        "@minizip_repo//:minizip",
        "@picosha2_repo//:picosha2",
        "@semver_repo//:semver",
    ],
    alwayslink = True,
)

neuropod_cc_library(
    name = "blocking_spsc_queue",
    hdrs = [
        "blocking_spsc_queue.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
)

# Package all the header files
pkg_tar(
    name = "libneuropod_internal_hdrs",
    srcs = glob(["*.hh"]),
    package_dir = "internal/",
    visibility = [
        "//visibility:public",
    ],
)
