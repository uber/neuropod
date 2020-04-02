#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
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
