#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("//bazel:copy_libs.bzl", "copy_libs")

cc_binary(
    name = "libneuropod_torchscript_backend.so",
    srcs = [
        "torch_backend.cc",
        "type_utils.cc",
        "//neuropod:libneuropod.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":neuropod_torchscript_backend_hdrs",
        "//neuropod/internal:deleter",
        "@libtorch_repo//:libtorch",
    ],
    data = [
        ":copy_libtorch",
    ]
)

cc_library(
    name = "neuropod_torchscript_backend_hdrs",
    hdrs = [
        "torch_backend.hh",
        "torch_tensor.hh",
        "type_utils.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends:neuropod_backend",
        "@libtorch_repo//:libtorch_hdrs",
    ]
)

copy_libs(
    name = "copy_libtorch",
    libs = "@libtorch_repo//:libtorch_libs"
)

pkg_tar(
    name = "neuropod_torchscript_backend",
    srcs = [
        ":libneuropod_torchscript_backend.so",
        "@libtorch_repo//:libtorch_libs",
    ],
    tags = ["manual"],
    extension = "tar.gz",
    visibility = [
        "//visibility:public",
    ],
)
