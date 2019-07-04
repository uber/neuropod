#
# Uber, Inc. (c) 2018
#

cc_binary(
    name = "libneuropod_pythonbridge_backend.so",
    srcs = [
        "python_bridge.cc",
        "python_bridge.hh",
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
        "//neuropods/backends:neuropod_backend",
        "//neuropods/backends/test_backend:test_backend_hdrs",
        "//neuropods/bindings:bindings",
        "//neuropods/internal:deleter",
    ],
)
