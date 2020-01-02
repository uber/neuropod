#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "serialization_hdrs",
    hdrs = [
        "serialization.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "@boost_repo//:boost",
    ]
)

cc_library(
    name = "serialization_impl",
    srcs = [
        "serialization.cc",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":serialization_hdrs",
        "//neuropods/backends:neuropod_backend",
    ],
)

# Package all the header files
pkg_tar(
    name = "libneuropods_serialization_hdrs",
    package_dir = "serialization/",
    srcs = glob(["*.hh"]),
    visibility = [
        "//visibility:public",
    ],
)
