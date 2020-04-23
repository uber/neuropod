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
    linkshared = True,
    deps = [
        ":bindings",
        "//neuropod/core",
        "//neuropod/multiprocess",
        "//neuropod/serialization",
    ],
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
        "//neuropod:neuropod_hdrs",
        "//neuropod/internal",
        "@pybind11_repo//:pybind11",
    ],
)
