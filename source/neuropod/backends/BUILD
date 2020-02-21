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
        "tensor_allocator.hh",
        "neuropod_backend.hh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//neuropod/internal:backend_registration",
        "//neuropod/internal:deleter",
        "//neuropod/internal:neuropod_config_utils",
        "//neuropod/internal:neuropod_tensor",
        "//neuropod/internal:neuropod_loader",
    ],
)

# Package all the header files
pkg_tar(
    name = "libneuropod_backends_hdrs",
    package_dir = "backends/",
    srcs = glob(["*.hh"]),
    visibility = [
        "//visibility:public",
    ],
)
