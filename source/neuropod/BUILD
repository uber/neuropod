#
# Uber, Inc. (c) 2018
#

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
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal:neuropod_config_utils",
        "//neuropod/internal:neuropod_tensor",
    ],
)

cc_binary(
    name = "libneuropod.so",
    linkshared = True,
    linkstatic = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    srcs = [
        "neuropod.cc",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":neuropod_hdrs",
        "//neuropod/internal:backend_registration_impl",
        "//neuropod/internal:neuropod_loader_impl",
        "//neuropod/internal:neuropod_tensor_raw_data_access_impl",
        "//neuropod/internal:tensor_serialization_impl",
        "//neuropod/multiprocess:multiprocess_impl",
        "//neuropod/serialization:serialization_impl",
    ]
)

pkg_tar(
    name = "libneuropod_hdrs",
    package_dir = "include/neuropod/",
    srcs = [
        # Headers
        ":neuropod.hh",
        ":version.hh",
    ],
    deps = [
        "//neuropod/internal:libneuropod_internal_hdrs",
        "//neuropod/backends:libneuropod_backends_hdrs",
        "//neuropod/serialization:libneuropod_serialization_hdrs",
        "//neuropod/multiprocess:libneuropod_multiprocess_hdrs",
    ],
)

pkg_tar(
    name = "libneuropod_libs",
    package_dir = "lib/",
    srcs = [
        ":libneuropod.so",
    ]
)

pkg_tar(
    name = "libneuropod_bins",
    package_dir = "bin/",
    srcs = [
        "//neuropod/multiprocess:neuropod_multiprocess_worker"
    ]
)

pkg_tar(
    name = "libneuropod",
    extension = "tar.gz",
    deps = [
        ":libneuropod_hdrs",
        ":libneuropod_libs",
        ":libneuropod_bins",
    ]
)

filegroup(
    name = "packages",
    tags = ["manual"],
    srcs = [
        "//neuropod/backends/python_bridge:neuropod_pythonbridge_backend",
        "//neuropod/backends/torchscript:neuropod_torchscript_backend",
        "//neuropod/backends/tensorflow:neuropod_tensorflow_backend",
    ],
)
