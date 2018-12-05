#
# Uber, Inc. (c) 2018
#

cc_test(
    name = "test_input_output",
    srcs = ["test_input_output.cc"],
    deps = [
        "@gtest//:main",
        "//neuropods",
        "//neuropods/backends/test_backend",
    ],
)

cc_test(
    name = "test_python_bridge",
    srcs = ["test_python_bridge.cc"],
    deps = [
        ":neuropods_test_utils",
        "//neuropods/backends/python_bridge",
    ],
)

cc_library(
    name = "neuropods_test_utils",
    hdrs = ["test_utils.hh"],
    deps = [
        "@gtest//:main",
        "//neuropods",
    ],
    data = [
        "//neuropods/tests/test_data"
    ],
)
