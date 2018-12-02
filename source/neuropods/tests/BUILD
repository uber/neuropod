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
    name = "test_models",
    srcs = ["test_models.cc"],
    deps = [
        "//neuropods",
        "//neuropods/backends/python_bridge",
    ],
)
