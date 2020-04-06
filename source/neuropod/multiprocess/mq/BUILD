#
# Uber, Inc. (c) 2020
#

cc_library(
    name = "mq",
    srcs = [
        "ipc_message_queue.cc",
        "transferrables.cc",
    ],
    hdrs = [
        "heartbeat.hh",
        "ipc_message_queue.hh",
        "ipc_message_queue_impl.hh",
        "transferrables.hh",
        "wire_format.hh",
        "wire_format_impl.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/internal:blocking_spsc_queue",
        "//neuropod/multiprocess/serialization",
        "//neuropod/multiprocess/shm",
        "@boost_repo//:boost",
    ],
)
