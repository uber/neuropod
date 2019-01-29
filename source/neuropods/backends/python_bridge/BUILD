#
# Uber, Inc. (c) 2018
#

cc_binary(
    name = "libneuropod_pythonbridge_backend.so",
    srcs = [
        "numpy_neuropod_tensor.hh",
        "python_bridge.cc",
        "python_bridge.hh",
        "type_utils.cc",
        "type_utils.hh",
        "//neuropods:libneuropods.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = ["-Wl,-rpath ."],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
        "@boost//:python",
        "@python_numpy_repo//:python_numpy",
    ]
)
