#
# Uber, Inc. (c) 2019
#

cc_binary(
    name = "neuropod_native.so",
    srcs = [
        "neuropod_native.cc",
        "//neuropod:libneuropod.so",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    deps = [
        ":bindings",
        "//neuropod/multiprocess",
        "//neuropod/serialization",
        "//neuropod/backends/test_backend:test_backend_hdrs",
    ],
    linkshared = True,
)

cc_library(
    name = "bindings",
    srcs = [
        "python_bindings.cc",
    ],
    hdrs = [
        "python_bindings.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/internal",
        "//neuropod:neuropod_hdrs",
        "@pybind11_repo//:pybind11",
    ],
)
