#
# Uber, Inc. (c) 2018
#

cc_binary(
    name = "libneuropod_torchscript_backend.so",
    srcs = [
        "torch_backend.cc",
        "torch_backend.hh",
        "torch_tensor.hh",
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
            "@libtorch_repo_darwin//:libtorch_hdrs",
            "@libtorch_repo_darwin//:libtorch",
        ],
        "//conditions:default": [
            "@libtorch_repo_linux//:libtorch_hdrs",
            "@libtorch_repo_linux//:libtorch",
        ],
    })
)
