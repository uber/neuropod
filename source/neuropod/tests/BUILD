#
# Uber, Inc. (c) 2018
#

cc_test(
    name = "benchmark_accessor",
    srcs = [
        "benchmark_accessor.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@benchmark//:benchmark_main",
        "//neuropod/backends/test_backend:test_backend_hdrs",
        "//neuropod:neuropod_hdrs",
    ],
)

cc_test(
    name = "benchmark_multiprocess",
    srcs = [
        "benchmark_multiprocess.cc",
        "//neuropod:libneuropod.so",
    ],
    deps = [
        "@benchmark//:benchmark_main",
        "//neuropod:neuropod_hdrs",
        "//neuropod/multiprocess:multiprocess_hdrs",
    ],
    data = [
        "//neuropod/tests/test_data",
        "//neuropod/backends/python_bridge:libneuropod_pythonbridge_backend.so",
        "//neuropod/backends/torchscript:libneuropod_torchscript_backend.so",
        "//neuropod/backends/tensorflow:libneuropod_tensorflow_backend.so",
    ],
)

cc_test(
    name = "benchmark_serialization",
    srcs = [
        "benchmark_serialization.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@benchmark//:benchmark_main",
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends/test_backend:test_backend_hdrs",
        "//neuropod/serialization:serialization_hdrs",
    ],
)

cc_test(
    name = "benchmark_shm_allocator",
    srcs = [
        "benchmark_shm_allocator.cc",
    ],
    deps = [
        "@benchmark//:benchmark_main",
        "//neuropod/multiprocess:shm_allocator",
    ],
)

cc_test(
    name = "test_input_output",
    srcs = [
        "test_input_output.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends/test_backend:test_backend_hdrs",
        "//neuropod/serialization:serialization_hdrs",
    ],
)

cc_test(
    name = "test_serialization",
    srcs = [
        "test_serialization.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends/test_backend:test_backend_hdrs",
        "//neuropod/serialization:serialization_hdrs",
    ],
)

cc_test(
    name = "test_python_bridge",
    srcs = [
        "test_python_bridge.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/python_bridge:libneuropod_pythonbridge_backend.so",
    ],
    deps = [
        ":neuropod_test_utils",
    ],
)

cc_test(
    name = "gpu_test_python_bridge",
    srcs = [
        "gpu_test_python_bridge.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/python_bridge:libneuropod_pythonbridge_backend.so",
    ],
    deps = [
        ":neuropod_test_utils",
    ],
)

cc_test(
    name = "test_torchscript_backend",
    srcs = [
        "test_torchscript_backend.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/torchscript:libneuropod_torchscript_backend.so",
    ],
    deps = [
        ":neuropod_test_utils",
        "//neuropod/backends/torchscript:neuropod_torchscript_backend_hdrs",
    ],
)

cc_test(
    name = "test_tensorflow_backend",
    srcs = [
        "test_tensorflow_backend.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/tensorflow:libneuropod_tensorflow_backend.so",
    ],
    deps = [
        ":neuropod_test_utils",
    ],
)

cc_test(
    name = "test_default_backend",
    srcs = [
        "test_default_backend.cc",
        "//neuropod:libneuropod.so",
    ],
    deps = [
        ":neuropod_test_utils",
    ],
    data = [
        "//neuropod/backends/python_bridge:libneuropod_pythonbridge_backend.so",
        "//neuropod/backends/torchscript:libneuropod_torchscript_backend.so",
        "//neuropod/backends/tensorflow:libneuropod_tensorflow_backend.so",
    ],
    linkstatic = True,
)

cc_library(
    name = "neuropod_test_utils",
    hdrs = ["test_utils.hh"],
    srcs = ["//neuropod:libneuropod.so"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_hdrs",
    ],
    data = [
        "//neuropod/tests/test_data"
    ],
)

cc_test(
    name = "test_config_utils",
    srcs = [
        "test_config_utils.cc",
        "//neuropod:libneuropod.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_hdrs",
        "//neuropod/internal:neuropod_config_utils",
    ],
)

cc_test(
    name = "test_accessor",
    srcs = [
        "test_accessor.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod/backends/test_backend:test_backend_hdrs",
        "//neuropod:neuropod_hdrs",
    ],
)

cc_test(
    name = "test_shm_allocator",
    srcs = [
        "test_shm_allocator.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod/multiprocess:shm_allocator",
    ],
)

cc_test(
    name = "test_shm_tensor",
    srcs = [
        "test_shm_tensor.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod/multiprocess:shm_tensor",
    ],
)

cc_test(
    name = "test_ipc_control_channel",
    srcs = [
        "test_ipc_control_channel.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod/multiprocess:ipc_control_channel",
    ],
)

cc_test(
    name = "test_multiprocess_worker",
    srcs = [
        "test_multiprocess_worker.cc",
        "//neuropod:libneuropod.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod/multiprocess:multiprocess_worker",
    ],
)

cc_test(
    name = "test_multiprocess_allowed_transitions",
    srcs = [
        "test_multiprocess_allowed_transitions.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod/multiprocess:ipc_control_channel",
    ],
)

cc_test(
    name = "test_multiprocess",
    srcs = [
        "test_multiprocess.cc",
        "//neuropod:libneuropod.so",
    ],
    deps = [
        ":neuropod_test_utils",
        "//neuropod/multiprocess:multiprocess_hdrs",
    ],
    data = [
        "//neuropod/backends/python_bridge:libneuropod_pythonbridge_backend.so",
        "//neuropod/backends/torchscript:libneuropod_torchscript_backend.so",
        "//neuropod/backends/tensorflow:libneuropod_tensorflow_backend.so",
    ],
)

cc_test(
    name = "test_internal_neuropod_tensor",
    srcs = [
        "test_internal_neuropod_tensor.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod/backends/test_backend:test_backend_hdrs",
        "//neuropod/internal:neuropod_tensor",
    ],
)

cc_test(
    name = "test_conversions_eigen",
    srcs = [
        "test_conversions_eigen.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends/test_backend:test_backend_hdrs",
        "//neuropod/conversions:eigen",
        "@eigen3_repo//:eigen3",
    ],
)

cc_test(
    name = "test_test_backend",
    srcs = [
        "test_test_backend.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends/test_backend:test_backend_hdrs",
    ],
)

cc_test(
    name = "test_backend_registration",
    srcs = [
        "test_backend_registration.cc",
        "//neuropod:libneuropod.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_hdrs",
        "//neuropod/internal:backend_registration",
    ],
)

cc_test(
    name = "test_factories",
    srcs = [
        "test_factories.cc",
        "//neuropod:libneuropod.so",
        "//neuropod/backends/test_backend:libneuropod_test_backend.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod/backends/test_backend:test_backend_hdrs",
    ],
)

cc_test(
    name = "test_loader",
    srcs = [
        "test_loader.cc",
        "//neuropod:libneuropod.so",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_hdrs",
        "//neuropod/internal:neuropod_loader",
    ],
    data = [
        "//neuropod/tests/test_data"
    ],
)
