#
# Uber, Inc. (c) 2019
#

cc_binary(
    name = "neuropods_native.so",
    srcs = [
        "python_bindings.cc",
        "//neuropods:libneuropods.so",
    ],
    deps = [
        "//neuropods/internal:neuropod_tensor",
        "//neuropods:neuropods_hdrs",
        "@pybind11_repo//:pybind11",
    ],
    linkshared = True,
)
