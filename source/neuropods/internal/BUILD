#
# Uber, Inc. (c) 2018
#

# All the targets in this folder are internal types
# This means we only want them to be accessible from
# within neuropods

cc_library(
    name = "neuropod_tensor",
    hdrs = [
        "neuropod_tensor.hh",
        "tensor_types.hh",
        "type_macros.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
)

cc_library(
    name = "tensor_store",
    srcs = [
        "tensor_store.cc",
    ],
    hdrs = [
        "tensor_store.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":neuropod_tensor",
    ],
)
