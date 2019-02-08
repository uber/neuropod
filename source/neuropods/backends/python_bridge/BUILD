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
        "@boost_repo//:boost",
    ] + select({
        "@bazel_tools//src/conditions:darwin": ["@python_numpy_repo_darwin//:python_numpy"],
        "//conditions:default": ["@python_numpy_repo_linux//:python_numpy"],
    })
)
