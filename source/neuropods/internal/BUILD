#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "neuropod_config_utils",
    srcs = [
        "config_utils.cc",
    ],
    hdrs = [
        "config_utils.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":neuropod_tensor",
        "@libjsoncpp_repo//:libjsoncpp",
    ],
)

cc_library(
    name = "neuropod_tensor",
    hdrs = [
        "neuropod_tensor.hh",
        "tensor_types.hh",
        "type_macros.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
)

cc_library(
    name = "tensor_store",
    srcs = [
        "tensor_store.cc",
    ],
    hdrs = [
        "tensor_store.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":neuropod_tensor",
    ],
)

cc_library(
    name = "backend_registration",
    hdrs = [
        "backend_registration.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":neuropod_config_utils",
    ],
)

cc_library(
    name = "backend_registration_impl",
    srcs = [
        "backend_registration.cc",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":backend_registration",
    ],
    linkopts = ["-ldl"],
)

# Package all the header files
pkg_tar(
    name = "libneuropods_internal_hdrs",
    package_dir = "internal/",
    srcs = glob(["*.hh"]),
    visibility = [
        "//visibility:public",
    ],
)
