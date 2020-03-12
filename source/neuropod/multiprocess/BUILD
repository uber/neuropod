#
# Uber, Inc. (c) 2019
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "shm_allocator",
    srcs = [
        "raw_shm_block_allocator.hh",
        "raw_shm_block_allocator.cc",
        "shm_allocator.hh",
        "shm_allocator.cc",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [],
        "//conditions:default": ["-lrt"],
    }),
    deps = [
        "//neuropod/internal:error_utils",
        "//neuropod/internal:logging",
        "//neuropod/internal:memory_utils",
        "@boost_repo//:boost",
    ],
)

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
        ":shm_allocator",
    ],
)

cc_library(
    name = "ipc_control_channel",
    hdrs = [
        "ipc_control_channel.hh",
    ],
    srcs = [
        "ipc_control_channel.cc",
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
        "control_messages.hh",
        "ipc_serialization.hh",
    ],
    srcs = [
        "ipc_message_queue.cc",
        "control_messages.cc",
        "ipc_serialization.cc",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":shm_tensor",
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
