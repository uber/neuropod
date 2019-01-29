#
# Uber, Inc. (c) 2018
#

# All the targets in this folder are internal types
# This means we only want them to be accessible from
# within neuropods

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "neuropod_proxy",
    srcs = [],
    hdrs = [
        "neuropod_proxy.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods:neuropods_fwd_declarations",
    ],
)
