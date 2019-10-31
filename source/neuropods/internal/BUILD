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
        ":error_utils",
        ":memory_utils",
        ":neuropod_loader",
        ":neuropod_tensor",
        "@libjsoncpp_repo//:libjsoncpp",
    ],
)

cc_library(
    name = "memory_utils",
    hdrs = [
        "memory_utils.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
)

cc_library(
    name = "logging",
    hdrs = [
        "logging.hh",
    ],
    srcs = [
        "logging.cc"
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
)

cc_library(
    name = "neuropod_tensor",
    srcs = [
        "neuropod_tensor.cc",
    ],
    hdrs = [
        "tensor_accessor.hh",
        "neuropod_tensor.hh",
        "tensor_types.hh",
        "type_macros.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":error_utils",
        ":memory_utils",
    ],
)

cc_library(
    name = "tensor_serialization_impl",
    srcs = [
        "neuropod_tensor_serialization.cc",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":neuropod_tensor_raw_data_access",
        "@boost_repo//:boost",
        "//neuropods/backends:neuropod_backend",
        "//neuropods/serialization:serialization_hdrs",
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
        ":error_utils",
        ":memory_utils",
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

cc_library(
    name = "deleter",
    hdrs = [
        "deleter.hh",
    ],
    srcs = [
        "deleter.cc",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
)

cc_library(
    name = "error_utils",
    hdrs = [
        "error_utils.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
)

cc_library(
    name = "neuropod_loader",
    hdrs = [
        "neuropod_loader.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
)

cc_library(
    name = "neuropod_loader_impl",
    srcs = [
        "neuropod_loader.cc",
    ],
    deps = [
        ":error_utils",
        ":neuropod_loader",
        ":memory_utils",
        "@filesystem_repo//:filesystem",
        "@picosha2_repo//:picosha2",
        "@zipper_repo//:zipper",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    linkopts = ["-lz"],
)

cc_library(
    name = "neuropod_tensor_raw_data_access",
    hdrs = [
        "neuropod_tensor_raw_data_access.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
)

cc_library(
    name = "neuropod_tensor_raw_data_access_impl",
    srcs = [
        "neuropod_tensor_raw_data_access.cc",
    ],
    deps = [
        ":neuropod_tensor",
        ":neuropod_tensor_raw_data_access",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
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
