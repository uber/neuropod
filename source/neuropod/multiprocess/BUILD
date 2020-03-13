#
# Uber, Inc. (c) 2019
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "shm_tensor",
    srcs = [
        "shm_tensor.cc",
        "shm_tensor.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal:deleter",
        "//neuropod/multiprocess/shm",
        "//neuropod/multiprocess/serialization",
    ],
)

cc_library(
    name = "ipc_control_channel",
    hdrs = [
        "ipc_control_channel.hh",
        "control_messages.hh",
        "ope_load_config.hh",
    ],
    srcs = [
        "ipc_control_channel.cc",
        "control_messages.cc",
        "ope_load_config.cc",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":ipc_message_queue",
    ],
)

cc_library(
    name = "ipc_message_queue",
    hdrs = [
        "ipc_message_queue.hh",
        "ipc_message_queue_impl.hh",
    ],
    srcs = [
        "ipc_message_queue.cc",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":shm_tensor",
        "//neuropod/internal:blocking_spsc_queue",
        "//neuropod/multiprocess/mq"
    ],
)

cc_library(
    name = "multiprocess_worker",
    hdrs = [
        "multiprocess_worker.hh",
    ],
    srcs = [
        "multiprocess_worker.cc",
        "tensor_utils.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":ipc_control_channel",
        "//neuropod:neuropod_hdrs",
        "//neuropod/internal:neuropod_tensor_raw_data_access",
    ],
)

cc_binary(
    name = "neuropod_multiprocess_worker",
    srcs = [
        "multiprocess_worker_main.cc",
        "//neuropod:libneuropod.so",
    ],
    linkstatic = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":multiprocess_worker",
    ],
)

cc_library(
    name = "multiprocess_hdrs",
    hdrs = [
        "multiprocess.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod:neuropod_hdrs",
    ],
    data = [
        ":neuropod_multiprocess_worker",
    ],
)

cc_library(
    name = "multiprocess_impl",
    srcs = [
        "multiprocess.cc",
    ],
    hdrs = [
        "multiprocess.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":ipc_control_channel",
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal:deleter",
        "@boost_repo//:boost",
    ],
)

# Package all the header files
pkg_tar(
    name = "libneuropod_multiprocess_hdrs",
    package_dir = "multiprocess/",
    srcs = glob(["*.hh"]),
    visibility = [
        "//visibility:public",
    ],
)
