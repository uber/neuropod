#
# Uber, Inc. (c) 2020
#

cc_library(
    name = "core",
    hdrs = glob(["*.hh"]),
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal",
    ],
)

cc_library(
    name = "impl",
    srcs = glob(["*.cc"]),
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":core",
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal",
    ],
    alwayslink = True,
)
