#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "neuropods_hdrs",
    hdrs = [
        "neuropod_input_builder.hh",
        "neuropod_output_data.hh",
        "neuropods.hh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":neuropods_fwd_declarations",
        "//neuropods/backends:neuropod_backend",
        "//neuropods/internal:neuropod_config_utils",
        "//neuropods/internal:neuropod_input_data",
        "//neuropods/internal:neuropod_tensor",
        "//neuropods/internal:tensor_store",
    ],
)

cc_library(
    name = "neuropods_fwd_declarations",
    srcs = [],
    hdrs = [
        "fwd_declarations.hh",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_binary(
    name = "libneuropods.so",
    linkshared = True,
    linkstatic = True,
    srcs = [
        "neuropod_input_builder.cc",
        "neuropod_output_data.cc",
        "neuropods.cc",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":neuropods_hdrs",
        "//neuropods/internal:backend_registration_impl",
    ]
)

pkg_tar(
    name = "libneuropods_hdrs",
    package_dir = "include/neuropods/",
    srcs = [
        # Headers
        ":neuropod_input_builder.hh",
        ":neuropod_output_data.hh",
        ":neuropods.hh",
        ":fwd_declarations.hh",
    ],
)

pkg_tar(
    name = "libneuropods_libs",
    package_dir = "lib/",
    srcs = [
        # Libs
        ":libneuropods.so",
        "//neuropods/backends/python_bridge:libneuropod_pythonbridge_backend.so",
        "//neuropods/backends/torchscript:libneuropod_torchscript_backend.so",
    ]
)

pkg_tar(
    name = "libneuropods",
    extension = "tar.gz",
    deps = [
        ":libneuropods_hdrs",
        ":libneuropods_libs",
    ]
)
