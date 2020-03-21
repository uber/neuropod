#
# Uber, Inc. (c) 2020
#

cc_library(
    name = "shm",
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
        "//neuropod/internal",
        "@boost_repo//:boost",
    ],
)
