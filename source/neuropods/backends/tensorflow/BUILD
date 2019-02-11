#
# Uber, Inc. (c) 2018
#

cc_binary(
    name = "libneuropod_tensorflow_backend.so",
    srcs = [
        "tf_backend.cc",
        "tf_backend.hh",
        "tf_tensor.hh",
        "tf_wrappers.hh",
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
    ] + select({
        "@bazel_tools//src/conditions:darwin": [
            "@tensorflow_repo_darwin//:tensorflow_c_hdrs",
            "@tensorflow_repo_darwin//:libtensorflow",
        ],
        "//conditions:default": [
            "@tensorflow_repo_linux//:tensorflow_c_hdrs",
            "@tensorflow_repo_linux//:libtensorflow",
        ],
    })
)
