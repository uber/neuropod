#
# Uber, Inc. (c) 2020
#

cc_library(
    name = "mq",
    hdrs = [
        "heartbeat.hh",
        "transferrables.hh",
        "wire_format.hh",
        "wire_format_impl.hh",
    ],
    srcs = [
        "wire_format.cc",
        "transferrables.cc",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/multiprocess/shm",
        "//neuropod/multiprocess/serialization",
        "@boost_repo//:boost",
    ],
)
