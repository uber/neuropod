#
# Uber, Inc. (c) 2018
#

# All the targets in this folder are internal types
# This means we only want them to be accessible from
# within neuropods

cc_library(
    name = "neuropod_config_utils",
    srcs = [
        "config_utils.cc",
    ],
    hdrs = [
        "config_utils.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/internal:neuropod_tensor",
        "@libjsoncpp_repo//:libjsoncpp",
    ],
)

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

cc_library(
    name = "neuropod_input_data",
    srcs = [
        "neuropod_input_data.cc",
    ],
    hdrs = [
        "neuropod_input_data.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":tensor_store",
        "//neuropods:neuropods_fwd_declarations",
    ],
)

cc_library(
    name = "backend_registration",
    srcs = [
        "backend_registration.cc",
    ],
    hdrs = [
        "backend_registration.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":neuropod_config_utils",
    ],
)
