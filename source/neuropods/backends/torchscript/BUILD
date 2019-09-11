#
# Uber, Inc. (c) 2018
#

load("//bazel:copy_libs.bzl", "copy_libs")

cc_binary(
    name = "libneuropod_torchscript_backend.so",
    srcs = [
        "torch_backend.cc",
        "torch_backend.hh",
        "torch_tensor.hh",
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
        "//neuropods:neuropods_hdrs",
        "//neuropods/backends:neuropod_backend",
        "//neuropods/internal:deleter",
        "@libtorch_repo//:libtorch_hdrs",
        "@libtorch_repo//:libtorch",
    ],
    data = [
        ":copy_libtorch",
    ]
)

copy_libs(
    name = "copy_libtorch",
    libs = "@libtorch_repo//:libtorch_libs"
)
