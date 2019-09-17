#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_binary(
    name = "libneuropod_pythonbridge_backend.so",
    srcs = [
        "python_bridge.cc",
        "python_bridge.hh",
        "//neuropods:libneuropods.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
        "-ldl",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
        "//neuropods/backends/test_backend:test_backend_hdrs",
        "//neuropods/bindings:bindings",
        "//neuropods/internal:deleter",
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
