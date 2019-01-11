#
# Uber, Inc. (c) 2018
#

cc_library(
    name = "torchscript_backend",
    srcs = [
        "torch_backend.cc",
        "torch_tensor.hh",
        "type_utils.cc",
        "type_utils.hh",
    ],
    hdrs = [
        "torch_backend.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
        "@libtorch_repo//:libtorch",
    ],
)
