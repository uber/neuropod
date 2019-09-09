#
# Uber, Inc. (c) 2019
#

cc_library(
    name = "shm_allocator",
    srcs = [
        "shm_allocator.hh",
        "shm_allocator.cc",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [],
        "//conditions:default": ["-lrt"],
    }),
    deps = [
        "//neuropods/internal:error_utils",
        "//neuropods/internal:logging",
        "//neuropods/internal:memory_utils",
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
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
        "//neuropods/internal:deleter",
        ":shm_allocator",
    ],
)

cc_library(
    name = "ipc_control_channel",
    hdrs = [
        "ipc_control_channel.hh",
        "control_messages.hh",
    ],
    srcs = [
        "ipc_control_channel.cc",
    ],
    visibility = [
        "//neuropods:__subpackages__",
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
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":ipc_control_channel",
        "//neuropods:neuropods_hdrs",
    ],
)

cc_binary(
    name = "neuropod_multiprocess_worker",
    srcs = [
        "multiprocess_worker_main.cc",
        "//neuropods:libneuropods.so",
    ],
    linkstatic = True,
    visibility = [
        "//neuropods:__subpackages__",
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
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods:neuropods_hdrs",
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
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":ipc_control_channel",
        "//neuropods:neuropods_hdrs",
        "//neuropods/backends:neuropod_backend",
        "//neuropods/internal:deleter",
        "@boost_repo//:boost",
    ],
)
