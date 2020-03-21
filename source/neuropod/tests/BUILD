#
# Uber, Inc. (c) 2018
#

cc_test(
    name = "benchmark_accessor",
    srcs = [
        "benchmark_accessor.cc",
    ],
    deps = [
        "@benchmark//:benchmark_main",
        "//neuropod/backends/test_backend",
        "//neuropod:neuropod_impl",
    ],
)

cc_test(
    name = "benchmark_multiprocess",
    srcs = [
        "benchmark_multiprocess.cc",
    ],
    deps = [
        "@benchmark//:benchmark_main",
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess",
        "//neuropod/backends/tensorflow:tensorflow_backend",
    ],
    data = [
        "//neuropod/tests/test_data",
    ],
)

cc_test(
    name = "benchmark_serialization",
    srcs = [
        "benchmark_serialization.cc",
    ],
    deps = [
        "@benchmark//:benchmark_main",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/test_backend",
        "//neuropod/serialization",
    ],
)

cc_test(
    name = "benchmark_shm_allocator",
    srcs = [
        "benchmark_shm_allocator.cc",
    ],
    deps = [
        "@benchmark//:benchmark_main",
        "//neuropod/multiprocess/shm",
    ],
)

cc_test(
    name = "test_input_output",
    srcs = [
        "test_input_output.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/test_backend",
        "//neuropod/serialization",
    ],
)

cc_test(
    name = "test_serialization",
    srcs = [
        "test_serialization.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/test_backend",
        "//neuropod/serialization",
    ],
)

cc_test(
    name = "test_ipc_serialization",
    srcs = [
        "test_ipc_serialization.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess/serialization",
        "//neuropod/multiprocess:ipc_control_channel",
        "//neuropod/multiprocess:shm_tensor",
    ],
)

cc_test(
    name = "test_ope_wire_format",
    srcs = [
        "test_ope_wire_format.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess/mq",
    ],
)

cc_test(
    name = "test_ope_heartbeat",
    srcs = [
        "test_ope_heartbeat.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess/mq",
    ],
)

cc_test(
    name = "test_ope_transferrables",
    srcs = [
        "test_ope_transferrables.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess/mq",
    ],
)

cc_test(
    name = "test_python_bridge",
    srcs = [
        "test_python_bridge.cc",
    ],
    deps = [
        ":neuropod_test_utils",
        "//neuropod/backends/python_bridge",
        "//neuropod:neuropod_impl",
    ],
)

cc_test(
    name = "gpu_test_python_bridge",
    srcs = [
        "gpu_test_python_bridge.cc",
    ],
    deps = [
        ":neuropod_test_utils",
        "//neuropod/backends/python_bridge",
        "//neuropod:neuropod_impl",
    ],
)

cc_test(
    name = "test_torchscript_backend",
    srcs = [
        "test_torchscript_backend.cc",
    ],
    deps = [
        ":neuropod_test_utils",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/torchscript:torch_backend",
    ],
)

cc_test(
    name = "test_tensorflow_backend",
    srcs = [
        "test_tensorflow_backend.cc",
    ],
    deps = [
        ":neuropod_test_utils",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/tensorflow:tensorflow_backend",
    ],
)

cc_test(
    name = "test_default_backend",
    srcs = [
        "test_default_backend.cc",
    ],
    deps = [
        ":neuropod_test_utils",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/torchscript:torch_backend",
        "//neuropod/backends/tensorflow:tensorflow_backend",
        "//neuropod/backends/python_bridge",
    ],
    linkstatic = True,
)

cc_library(
    name = "neuropod_test_utils",
    hdrs = ["test_utils.hh"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
    ],
    data = [
        "//neuropod/tests/test_data"
    ],
)

cc_test(
    name = "test_config_utils",
    srcs = [
        "test_config_utils.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/internal",
    ],
)

cc_test(
    name = "test_accessor",
    srcs = [
        "test_accessor.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod/backends/test_backend",
        "//neuropod:neuropod_impl",
    ],
)

cc_test(
    name = "test_shm_allocator",
    srcs = [
        "test_shm_allocator.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess/shm",
    ],
)

cc_test(
    name = "test_shm_tensor",
    srcs = [
        "test_shm_tensor.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
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
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess:ipc_control_channel",
    ],
)

cc_test(
    name = "test_ipc_message_queue",
    srcs = [
        "test_ipc_message_queue.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess/mq",
        "//neuropod/multiprocess:ipc_control_channel",
    ],
)

cc_test(
    name = "test_multiprocess_worker",
    srcs = [
        "test_multiprocess_worker.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
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
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess:ipc_control_channel",
    ],
)

cc_test(
    name = "test_multiprocess",
    srcs = [
        "test_multiprocess.cc",
    ],
    deps = [
        ":neuropod_test_utils",
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess",
        "//neuropod/backends/torchscript:torch_backend",
        "//neuropod/backends/tensorflow:tensorflow_backend",
        "//neuropod/backends/python_bridge",
    ],
)

cc_test(
    name = "test_internal_neuropod_tensor",
    srcs = [
        "test_internal_neuropod_tensor.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/test_backend",
        "//neuropod/internal",
    ],
)

cc_test(
    name = "test_conversions_eigen",
    srcs = [
        "test_conversions_eigen.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/test_backend",
        "//neuropod/conversions:eigen",
        "@eigen3_repo//:eigen3",
    ],
)

cc_test(
    name = "test_test_backend",
    srcs = [
        "test_test_backend.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/test_backend",
    ],
)

cc_test(
    name = "test_backend_registration",
    srcs = [
        "test_backend_registration.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/internal",
    ],
)

cc_test(
    name = "test_factories",
    srcs = [
        "test_factories.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/test_backend",
    ],
)

cc_test(
    name = "test_loader",
    srcs = [
        "test_loader.cc",
    ],
    deps = [
        "@gtest//:main",
        "//neuropod:neuropod_impl",
        "//neuropod/internal",
    ],
    data = [
        "//neuropod/tests/test_data"
    ],
)
