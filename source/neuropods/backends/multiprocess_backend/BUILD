#
# Uber, Inc. (c) 2019
#

cc_binary(
    name = "libneuropod_multiprocess_backend.so",
    srcs = [
        "multiprocess_backend.cc",
        "shm_tensor.hh",
        "//neuropods:libneuropods.so",
    ],
    linkshared = True,
    linkstatic = True,
    linkopts = [
        "-Wl,-rpath .",
        "-lrt",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods/backends:neuropod_backend",
        "//neuropods/internal:deleter",
        "//neuropods/internal:serialized_tensor",
        "@boost_repo//:boost",
    ],
    data = [
        ":neuropod_multiprocess_worker",
    ],
)

cc_binary(
    name = "neuropod_multiprocess_worker",
    srcs = [
        "multiprocess_worker.cc",
        "shm_tensor.hh",
        "//neuropods:libneuropods.so",
    ],
    linkstatic = True,
    linkopts = [
        "-Wl,-rpath .",
        "-lrt",
    ],
    visibility = [
        "//neuropods:__subpackages__",
    ],
    deps = [
        "//neuropods:neuropods_hdrs",
        "//neuropods/backends:neuropod_backend",
        "//neuropods/internal:deleter",
        "//neuropods/internal:serialized_tensor",
        "//neuropods/internal:tensor_utils",
        "@boost_repo//:boost",
    ],
)
