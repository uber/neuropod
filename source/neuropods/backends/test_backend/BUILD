#
# Uber, Inc. (c) 2018
#

cc_library(
    name = "test_backend",
    srcs = [
        "test_neuropod_backend.cc",
        "test_neuropod_tensor.cc",
    ],
    hdrs = [
        "test_neuropod_backend.hh",
        "test_neuropod_tensor.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
    ],
)
