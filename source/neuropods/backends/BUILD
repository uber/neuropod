#
# Uber, Inc. (c) 2018
#

# All the targets in this folder are internal types
# This means we only want them to be accessible from
# within neuropods

cc_library(
    name = "neuropod_backend",
    srcs = [],
    hdrs = [
        "neuropod_backend.hh",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//neuropods/internal:backend_registration",
        "//neuropods/internal:neuropod_tensor",
        "//neuropods/internal:tensor_store",
    ],
)
