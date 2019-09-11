#
# Uber, Inc. (c) 2018
#

load("//bazel:copy_libs.bzl", "copy_libs")

cc_binary(
    name = "libneuropod_tensorflow_backend.so",
    srcs = [
        "tf_backend.cc",
        "tf_backend.hh",
        "tf_tensor.hh",
        "tf_wrappers.hh",
        "type_utils.cc",
        "type_utils.hh",
        "//neuropods:libneuropods.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
        "//neuropods/internal:deleter",
        "@tensorflow_repo//:tensorflow_c_hdrs",
        "@tensorflow_repo//:libtensorflow",
    ],
    data = [
        "copy_libtensorflow",
    ]
)

copy_libs(
    name = "copy_libtensorflow",
    libs = "@tensorflow_repo//:libtensorflow_libs"
)
