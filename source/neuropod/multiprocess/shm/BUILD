#
# Uber, Inc. (c) 2019
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

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
        "//neuropod/internal:error_utils",
        "//neuropod/internal:logging",
        "//neuropod/internal:memory_utils",
        "@boost_repo//:boost",
    ],
)