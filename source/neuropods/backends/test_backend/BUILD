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
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
    ],
)

cc_binary(
    name = "libneuropod_test_backend.so",
    srcs = [
        "test_neuropod_backend.cc",
        "//neuropods:libneuropods.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = ["-Wl,-rpath ."],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":test_backend_hdrs"
    ]
)
