#
# Uber, Inc. (c) 2019
#

cc_binary(
    name = "neuropods_native.so",
    srcs = [
        "neuropods_native.cc",
        "//neuropods:libneuropods.so",
    ],
    deps = [
        ":bindings",
        "//neuropods/serialization:serialization_hdrs",
        "//neuropods/backends/test_backend:test_backend_hdrs",
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
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/internal:neuropod_tensor",
        "//neuropods:neuropods_hdrs",
        "@pybind11_repo//:pybind11",
    ],
)
