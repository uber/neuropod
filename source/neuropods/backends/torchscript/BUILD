#
# Uber, Inc. (c) 2018
#

cc_library(
    name = "torchscript_backend_hdrs",
    hdrs = [
        "torch_backend.hh",
        "torch_tensor.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
        "@libtorch_repo//:libtorch_hdrs",
        "@libtorch_repo//:libtorch",
    ],
)

cc_binary(
    name = "libneuropod_torchscript_backend.so",
    srcs = [
        "torch_backend.cc",
        "type_utils.cc",
        "type_utils.hh",
        "//neuropods:libneuropods.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = ["-Wl,-rpath ."],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":torchscript_backend_hdrs",
    ]
)
