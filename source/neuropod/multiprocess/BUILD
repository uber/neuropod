#
# Uber, Inc. (c) 2019
#

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
        ":ipc_control_channel",
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal",
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
        "//neuropod/multiprocess/mq",
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
        ":shm_tensor",
        ":ipc_control_channel",
        "//neuropod:neuropod_hdrs",
        "//neuropod/internal",
        "@tbb_repo_linux//:tbb",
    ],
)

cc_binary(
    name = "neuropod_multiprocess_worker",
    srcs = [
        "multiprocess_worker_main.cc",
        "//neuropod:libneuropod.so",
    ],
    linkstatic = True,
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":multiprocess_worker",
    ],
)

cc_library(
    name = "multiprocess",
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
    name = "impl",
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
        ":shm_tensor",
        ":ipc_control_channel",
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal",
        "@boost_repo//:boost",
    ],
)
