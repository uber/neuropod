#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_binary(
    name = "libneuropod_pythonbridge_backend.so",
    srcs = [
        "python_bridge.cc",
        "python_bridge.hh",
        "//neuropod:libneuropod.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
        "-ldl",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/backends:neuropod_backend",
        "//neuropod/backends/test_backend:test_backend_hdrs",
        "//neuropod/bindings:bindings",
        "//neuropod/internal:deleter",
    ],
)

pkg_tar(
    name = "neuropod_pythonbridge_backend",
    srcs = [
        ":libneuropod_pythonbridge_backend.so",
    ],
    tags = ["manual"],
    extension = "tar.gz",
    visibility = [
        "//visibility:public",
    ],
)
