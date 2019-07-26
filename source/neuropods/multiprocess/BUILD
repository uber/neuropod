#
# Uber, Inc. (c) 2019
#

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
        "control_messages.hh",
        "message_utils.hh",
        "shm_tensor.hh",
        "shm_allocator.hh",
        "shm_allocator.cc",
    ],
    hdrs = [
        "multiprocess.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [],
        "//conditions:default": ["-lrt"],
    }),
    deps = [
        "//neuropods:neuropods_hdrs",
        "//neuropods/backends:neuropod_backend",
        "//neuropods/internal:deleter",
        "@boost_repo//:boost",
    ],
)

cc_binary(
    name = "neuropod_multiprocess_worker",
    srcs = [
        "multiprocess_worker.cc",
        "tensor_utils.hh",
        "//neuropods:libneuropods.so",
    ],
    linkstatic = True,
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":multiprocess_impl",
    ],
)
