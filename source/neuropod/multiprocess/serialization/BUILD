#
# Uber, Inc. (c) 2019
#

cc_library(
    name = "serialization",
    hdrs = [
        "ipc_serialization.hh",
    ],
    srcs = [
        "ipc_serialization.cc",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/multiprocess/shm",
        "//neuropod/backends:neuropod_backend",
        "@boost_repo//:boost",
    ],
)