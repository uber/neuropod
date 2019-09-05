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
    name = "message_utils",
    hdrs = [
        "message_utils.hh",
        "control_messages.hh",
    ],
    srcs = [
        "message_utils.cc",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":shm_tensor",
    ],
)
