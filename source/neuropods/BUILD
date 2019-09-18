#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "neuropods_hdrs",
    hdrs = [
        "neuropods.hh",
        "version.hh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
        "//neuropods/internal:neuropod_config_utils",
        "//neuropods/internal:neuropod_tensor",
    ],
)

cc_binary(
    name = "libneuropods.so",
    linkshared = True,
    linkstatic = True,
    srcs = [
        "neuropods.cc",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":neuropods_hdrs",
        "//neuropods/internal:backend_registration_impl",
        "//neuropods/internal:tensor_serialization_impl",
        "//neuropods/multiprocess:multiprocess_impl",
        "//neuropods/serialization:serialization_impl",
    ]
)

pkg_tar(
    name = "libneuropods_hdrs",
    package_dir = "include/neuropods/",
    srcs = [
        # Headers
        ":neuropods.hh",
        ":version.hh",
    ],
    deps = [
        "//neuropods/internal:libneuropods_internal_hdrs",
        "//neuropods/backends:libneuropods_backends_hdrs",
        "//neuropods/serialization:libneuropods_serialization_hdrs",
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
        "//neuropods/backends/tensorflow:libneuropod_tensorflow_backend.so",
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
