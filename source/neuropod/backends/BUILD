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
    name = "neuropod_backend",
    srcs = [
        "neuropod_backend.cc",
    ],
    hdrs = [
        "neuropod_backend.hh",
        "tensor_allocator.hh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//neuropod/internal",
    ],
)

# Package all the header files
pkg_tar(
    name = "libneuropod_backends_hdrs",
    srcs = glob(["*.hh"]),
    package_dir = "backends/",
    visibility = [
        "//visibility:public",
    ],
)
