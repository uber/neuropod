#
# Uber, Inc. (c) 2020
#

cc_library(
    name = "mq",
    hdrs = [
        "heartbeat.hh",
        "ipc_message_queue.hh",
        "ipc_message_queue_impl.hh",
        "transferrables.hh",
        "wire_format.hh",
        "wire_format_impl.hh",
    ],
    srcs = [
        "ipc_message_queue.cc",
        "wire_format.cc",
        "transferrables.cc",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/internal:blocking_spsc_queue",
        "//neuropod/multiprocess/shm",
        "//neuropod/multiprocess/serialization",
        "@boost_repo//:boost",
    ],
)
