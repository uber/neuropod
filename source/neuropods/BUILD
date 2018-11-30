#
# Uber, Inc. (c) 2018
#

cc_library(
    name = "neuropods",
    srcs = [
        "neuropod_input_builder.cc",
        "neuropod_output_data.cc",
        "neuropods.cc",
    ],
    hdrs = [
        "neuropod_input_builder.hh",
        "neuropod_output_data.hh",
        "neuropods.hh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
        "//neuropods/internal:neuropod_tensor",
        "//neuropods/internal:tensor_store",
        "//neuropods/proxy:neuropod_proxy",
    ],
)
