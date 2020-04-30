#
# Uber, Inc. (c) 2020
#

cc_library(
    name = "shm",
    srcs = [
        "pool_shm_allocator.cc",
        "pool_shm_allocator.hh",
        "shm_allocator.cc",
        "shm_allocator.hh",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [],
        "//conditions:default": ["-lrt"],
    }),
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/internal",
        "@boost_repo//:boost",
    ],
)
