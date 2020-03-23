#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_binary(
    name = "libneuropod_pythonbridge_backend.so",
    srcs = [
        "//neuropod:libneuropod.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":python_bridge",
    ],
)

cc_library(
    name = "python_bridge",
    srcs = [
        "python_bridge.cc",
        "python_bridge.hh",
    ],
    linkopts = [
        "-ldl",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/backends:neuropod_backend",
        "//neuropod/backends/test_backend:test_backend_hdrs",
        "//neuropod/bindings",
        "//neuropod/internal",
    ],
    alwayslink = True,
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
