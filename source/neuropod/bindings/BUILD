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

cc_binary(
    name = "neuropod_native.so",
    srcs = [
        "neuropod_native.cc",
        "//neuropod:libneuropod.so",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkshared = True,
    deps = [
        ":bindings",
        "//neuropod/core",
        "//neuropod/multiprocess",
        "//neuropod/serialization",
    ],
)

cc_library(
    name = "bindings",
    srcs = [
        "python_bindings.cc",
    ],
    hdrs = [
        "python_bindings.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod:neuropod_hdrs",
        "//neuropod/internal",
        "@pybind11_repo//:pybind11",
    ],
)
