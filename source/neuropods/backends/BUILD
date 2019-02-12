#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "neuropod_backend",
    srcs = [],
    hdrs = [
        "neuropod_backend.hh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//neuropods/internal:backend_registration",
        "//neuropods/internal:neuropod_tensor",
        "//neuropods/internal:tensor_store",
    ],
)

# Package all the header files
pkg_tar(
    name = "libneuropods_backends_hdrs",
    package_dir = "backends/",
    srcs = glob(["*.hh"]),
    visibility = [
        "//visibility:public",
    ],
)
