#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("//bazel:copy_libs.bzl", "copy_libs")

cc_binary(
    name = "libneuropod_tensorflow_backend.so",
    srcs = [
        "//neuropod:libneuropod.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":tensorflow_backend",
    ],
    data = [
        "copy_libtensorflow",
    ]
)

cc_library(
    name = "tensorflow_backend",
    srcs = [
        "tf_backend.cc",
        "tf_backend.hh",
        "tf_tensor.cc",
        "tf_tensor.hh",
        "type_utils.cc",
        "type_utils.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal",
        "@tensorflow_hdrs_repo//:tensorflow_hdrs",
        "@tensorflow_repo//:libtensorflow",
        "@libjsoncpp_repo//:libjsoncpp",
    ],
    alwayslink = True,
)

copy_libs(
    name = "copy_libtensorflow",
    libs = "@tensorflow_repo//:libtensorflow_libs"
)

pkg_tar(
    name = "neuropod_tensorflow_backend",
    srcs = [
        ":libneuropod_tensorflow_backend.so",
        "@tensorflow_repo//:libtensorflow_libs",
    ],
    tags = ["manual"],
    extension = "tar.gz",
    visibility = [
        "//visibility:public",
    ],
)
