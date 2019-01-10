#
# Uber, Inc. (c) 2018
#

cc_library(
    name = "python_bridge",
    srcs = [
        "numpy_neuropod_tensor.cc",
        "python_bridge.cc",
        "type_utils.cc",
        "type_utils.hh",
    ],
    hdrs = [
        "numpy_neuropod_tensor.hh",
        "python_bridge.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
        "@boost//:python",
        "@python_numpy_repo//:python_numpy",
    ],
)
