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
        "//neuropods/serialization:serialization_hdrs",
    ],
)

cc_test(
    name = "test_serialization",
    srcs = [
        "test_serialization.cc",
        "//neuropods:libneuropods.so",
        "//neuropods/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropods:neuropods_hdrs",
        "//neuropods/backends/test_backend:test_backend_hdrs",
        "//neuropods/serialization:serialization_hdrs",
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
    name = "gpu_test_python_bridge",
    srcs = [
        "gpu_test_python_bridge.cc",
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
        "//neuropods/backends/torchscript:neuropod_torchscript_backend_hdrs",
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
    data = [
        "//neuropods/backends/python_bridge:libneuropod_pythonbridge_backend.so",
        "//neuropods/backends/torchscript:libneuropod_torchscript_backend.so",
        "//neuropods/backends/tensorflow:libneuropod_tensorflow_backend.so",
    ],
    linkstatic = True,
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

cc_library(
    name = "timing_utils",
    hdrs = ["timing_utils.hh"],
    visibility = [
        "//visibility:public",
    ],
)

cc_test(
    name = "test_accessor",
    srcs = [
        "test_accessor.cc",
        "//neuropods:libneuropods.so",
        "//neuropods/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        ":timing_utils",
        "@gtest//:main",
        "//neuropods/backends/test_backend:test_backend_hdrs",
        "//neuropods:neuropods_hdrs",
    ],
)

cc_test(
    name = "test_shm_allocator",
    srcs = [
        "test_shm_allocator.cc",
    ],
    deps = [
        ":timing_utils",
        "@gtest//:main",
        "@benchmark//:benchmark",
        "//neuropods/multiprocess:shm_allocator",
    ],
)

cc_test(
    name = "test_shm_tensor",
    srcs = [
        "test_shm_tensor.cc",
    ],
    deps = [
        ":timing_utils",
        "@gtest//:main",
        "//neuropods/multiprocess:shm_tensor",
    ],
)

cc_test(
    name = "test_ipc_control_channel",
    srcs = [
        "test_ipc_control_channel.cc",
    ],
    deps = [
        ":timing_utils",
        "@gtest//:main",
        "//neuropods/multiprocess:ipc_control_channel",
    ],
)

cc_test(
    name = "test_multiprocess_worker",
    srcs = [
        "test_multiprocess_worker.cc",
        "//neuropods:libneuropods.so",
    ],
    deps = [
        ":timing_utils",
        "@gtest//:main",
        "//neuropods/multiprocess:multiprocess_worker",
    ],
)

cc_test(
    name = "test_multiprocess_allowed_transitions",
    srcs = [
        "test_multiprocess_allowed_transitions.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropods/multiprocess:ipc_control_channel",
    ],
)

cc_test(
    name = "test_multiprocess",
    srcs = [
        "test_multiprocess.cc",
        "//neuropods:libneuropods.so",
    ],
    deps = [
        ":neuropods_test_utils",
        "//neuropods/multiprocess:multiprocess_hdrs",
    ],
    data = [
        "//neuropods/backends/python_bridge:libneuropod_pythonbridge_backend.so",
        "//neuropods/backends/torchscript:libneuropod_torchscript_backend.so",
        "//neuropods/backends/tensorflow:libneuropod_tensorflow_backend.so",
    ],
)

cc_test(
    name = "test_multiprocess_perf",
    srcs = [
        "test_multiprocess_perf.cc",
        "//neuropods:libneuropods.so",
    ],
    deps = [
        ":neuropods_test_utils",
        ":timing_utils",
        "//neuropods/multiprocess:multiprocess_hdrs",
    ],
    data = [
        "//neuropods/backends/python_bridge:libneuropod_pythonbridge_backend.so",
        "//neuropods/backends/torchscript:libneuropod_torchscript_backend.so",
        "//neuropods/backends/tensorflow:libneuropod_tensorflow_backend.so",
    ],
)

cc_test(
    name = "test_internal_neuropod_tensor",
    srcs = [
        "test_internal_neuropod_tensor.cc",
        "//neuropods:libneuropods.so",
        "//neuropods/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropods/backends/test_backend:test_backend_hdrs",
        "//neuropods/internal:neuropod_tensor",
    ],
)

cc_test(
    name = "test_conversions_eigen",
    srcs = [
        "test_conversions_eigen.cc",
        "//neuropods:libneuropods.so",
        "//neuropods/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropods:neuropods_hdrs",
        "//neuropods/backends/test_backend:test_backend_hdrs",
        "//neuropods/conversions:eigen",
        "@eigen3_repo//:eigen3",
    ],
)

cc_test(
    name = "test_test_backend",
    srcs = [
        "test_test_backend.cc",
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
    name = "test_logging",
    srcs = [
        "test_logging.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropods/internal:logging",
    ],
)

cc_test(
    name = "test_backend_registration",
    srcs = [
        "test_backend_registration.cc",
        "//neuropods:libneuropods.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropods:neuropods_hdrs",
        "//neuropods/internal:backend_registration",
    ],
)

cc_test(
    name = "test_factories",
    srcs = [
        "test_factories.cc",
        "//neuropods:libneuropods.so",
        "//neuropods/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropods/backends/test_backend:test_backend_hdrs",
    ],
)
