#
# Uber, Inc. (c) 2018
#

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
    ]
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
    package_dir = "serialization/",
    srcs = glob(["*.hh"]),
    visibility = [
        "//visibility:public",
    ],
)
