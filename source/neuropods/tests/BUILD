#
# Uber, Inc. (c) 2018
#

cc_test(
    name = "test_input_output",
    srcs = [
        "test_input_output.cc",
        "//neuropods:libneuropods.so",
        "//neuropods/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropods:neuropods_hdrs",
        "//neuropods/backends/test_backend:test_backend_hdrs",
    ],
)

cc_test(
    name = "test_python_bridge",
    srcs = [
        "test_python_bridge.cc",
        "//neuropods:libneuropods.so",
        "//neuropods/backends/python_bridge:libneuropod_pythonbridge_backend.so",
    ],
    deps = [
        ":neuropods_test_utils",
    ],
)

cc_test(
    name = "test_torchscript_backend",
    srcs = [
        "test_torchscript_backend.cc",
        "//neuropods:libneuropods.so",
        "//neuropods/backends/torchscript:libneuropod_torchscript_backend.so",
    ],
    deps = [
        ":neuropods_test_utils",
    ],
)

cc_test(
    name = "test_tensorflow_backend",
    srcs = [
        "test_tensorflow_backend.cc",
        "//neuropods:libneuropods.so",
        "//neuropods/backends/tensorflow:libneuropod_tensorflow_backend.so",
    ],
    deps = [
        ":neuropods_test_utils",
    ],
)

cc_test(
    name = "test_default_backend",
    srcs = [
        "test_default_backend.cc",
        "//neuropods:libneuropods.so",
    ],
    deps = [
        ":neuropods_test_utils",
    ],
)

cc_library(
    name = "neuropods_test_utils",
    hdrs = ["test_utils.hh"],
    srcs = ["//neuropods:libneuropods.so"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@gtest//:main",
        "//neuropods:neuropods_hdrs",
    ],
    data = [
        "//neuropods/tests/test_data"
    ],
)

cc_test(
    name = "test_config_utils",
    srcs = [
        "test_config_utils.cc",
        "//neuropods:libneuropods.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropods:neuropods_hdrs",
        "//neuropods/internal:neuropod_config_utils",
    ],
)
