#
# Uber, Inc. (c) 2018
#

cc_binary(
    name = "libneuropod_pythonbridge_backend.so",
    srcs = [
        "//neuropods:libneuropods.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = [
        "-Wl,-rpath .",
        "-ldl",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        ":pythonbridge_hdrs",
        "//neuropods/internal:deleter",
        "@boost_repo//:boost",
    ] + select({
        "@bazel_tools//src/conditions:darwin": ["@python_numpy_repo_darwin//:python_numpy"],
        "//conditions:default": ["@python_numpy_repo_linux//:python_numpy"],
    })
)

cc_library(
    name = "pythonbridge",
    srcs = [
        "python_bridge.cc",
        "type_utils.cc",        
    ],
    hdrs = [
        "type_utils.hh",
        "numpy_neuropod_tensor.hh",
        "python_bridge.hh",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [ 
        "//neuropods/backends:neuropod_backend",
        "@boost_repo//:boost",
    ] + select({
        "@bazel_tools//src/conditions:darwin": ["@python_numpy_repo_darwin//:python_numpy"],
        "//conditions:default": ["@python_numpy_repo_linux//:python_numpy"],
    }),
)
