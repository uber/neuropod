#
# Uber, Inc. (c) 2018
#

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "internal",
    hdrs = glob(["*.hh"]),
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "@fmt_repo//:fmt",
        "@spdlog_repo//:spdlog",
    ],
)

cc_library(
    name = "impl",
    srcs = glob(["*.cc"]),
    linkopts = ["-ldl"],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":internal",
        "//neuropod/backends:neuropod_backend",
        "//neuropod/serialization",
        "@boost_repo//:boost",
        "@filesystem_repo//:filesystem",
        "@libjsoncpp_repo//:libjsoncpp",
        "@picosha2_repo//:picosha2",
        "@semver_repo//:semver",
        "@zipper_repo//:zipper",
    ],
    alwayslink = True,
)

cc_library(
    name = "blocking_spsc_queue",
    hdrs = [
        "blocking_spsc_queue.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
)

# Package all the header files
pkg_tar(
    name = "libneuropod_internal_hdrs",
    srcs = glob(["*.hh"]),
    package_dir = "internal/",
    visibility = [
        "//visibility:public",
    ],
)
