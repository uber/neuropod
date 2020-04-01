#
# Uber, Inc. (c) 2019
#

cc_binary(
    name = "neuropod_native.so",
    srcs = [
        "neuropod_native.cc",
        "//neuropod:libneuropod.so",
    ],
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
