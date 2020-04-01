#
# Uber, Inc. (c) 2018
#

cc_library(
    name = "test_backend_hdrs",
    hdrs = [
        "test_neuropod_backend.hh",
        "test_neuropod_tensor.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal",
    ],
)

cc_library(
    name = "test_backend",
    srcs = [
        "test_neuropod_backend.cc",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":test_backend_hdrs"
    ],
    alwayslink = True
)

cc_binary(
    name = "libneuropod_test_backend.so",
    srcs = [
        "//neuropod:libneuropod.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = ["-Wl,-rpath ."],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":test_backend"
    ]
)
