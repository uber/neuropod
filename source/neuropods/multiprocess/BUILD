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
