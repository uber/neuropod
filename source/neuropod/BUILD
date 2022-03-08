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
    name = "neuropod_hdrs",
    hdrs = [
        "neuropod.hh",
        "version.hh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":options",
        "//neuropod/backends:neuropod_backend",
        "//neuropod/core",
        "//neuropod/internal",
    ],
)

cc_library(
    name = "options",
    hdrs = ["options.hh"],
    visibility = [
        "//visibility:public",
    ],
)

cc_binary(
    name = "libneuropod_orig.so",
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkshared = True,
    linkstatic = True,
    deps = [
        ":neuropod_impl",
    ],
)

# This genrule changes the dylib id for libneuropod.so on mac
genrule(
    name = "change_dylib_id",
    srcs = [":libneuropod_orig.so"],
    outs = ["libneuropod.so"],
    cmd = select({
        "@bazel_tools//src/conditions:darwin": "cp -f $< $@; chmod 755 $@; install_name_tool -id @rpath/libneuropod.so $@",
        "//conditions:default": "cp -f $< $@",
    }),
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "neuropod_impl",
    srcs = [
        "neuropod.cc",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":neuropod_hdrs",
        "//neuropod/core:impl",
        "//neuropod/internal:impl",
        "//neuropod/multiprocess:impl",
        "//neuropod/serialization:impl",
    ],
    alwayslink = True,
)

pkg_tar(
    name = "libneuropod_hdrs",
    srcs = [
        # Headers
        ":neuropod.hh",
        ":version.hh",
        ":options.hh",
    ],
    package_dir = "include/neuropod/",
    deps = [
        "//neuropod/backends:libneuropod_backends_hdrs",
        "//neuropod/core:libneuropod_core_hdrs",
        "//neuropod/internal:libneuropod_internal_hdrs",
        "//neuropod/serialization:libneuropod_serialization_hdrs",
    ],
)

pkg_tar(
    name = "libneuropod_libs",
    srcs = [
        ":libneuropod.so",
    ],
    package_dir = "lib/",
)

pkg_tar(
    name = "libneuropod_bins",
    srcs = [
        "//neuropod/multiprocess:neuropod_multiprocess_worker",
    ],
    package_dir = "bin/",
)

pkg_tar(
    name = "libneuropod",
    extension = "tar.gz",
    deps = [
        ":libneuropod_bins",
        ":libneuropod_hdrs",
        ":libneuropod_libs",
    ],
)

filegroup(
    name = "packages",
    srcs = [
        "//neuropod/backends/python_bridge:neuropod_pythonbridge_backend",
        "//neuropod/backends/tensorflow:neuropod_tensorflow_backend",
        "//neuropod/backends/torchscript:neuropod_torchscript_backend",
    ],
    tags = ["manual"],
)
