#
# Uber, Inc. (c) 2018
#

cc_library(
    name = "stream",
    hdrs = [
        "stream.hh",
    ],
    deps = [
        "//neuropods:neuropods_hdrs",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
)

cc_library(
    name = "numpy",
    srcs = [
        "numpy.cc",
    ],
    hdrs = [
        "numpy.hh",
    ],
    deps = [
        "@boost_repo//:boost",
        "//neuropods/backends/python_bridge:pythonbridge",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
)
