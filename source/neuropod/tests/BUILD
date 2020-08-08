# Copyright (c) 2020 UATC, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cc_test(
    name = "benchmark_accessor",
    srcs = [
        "benchmark_accessor.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "@benchmark//:benchmark_main",
    ],
)

cc_test(
    name = "benchmark_multiprocess",
    srcs = [
        "benchmark_multiprocess.cc",
    ],
    data = [
        "//neuropod/backends/tensorflow:libneuropod_tensorflow_backend.so",
        "//neuropod/tests/test_data",
    ],
    tags = [
        "requires_path",
        # Don't do trace logging in CI for this benchmark
        "no_trace_logging",
    ],
    deps = [
        ":ope_overrides",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/tensorflow:tensorflow_backend",
        "//neuropod/multiprocess",
        "@benchmark//:benchmark_main",
    ],
)

cc_test(
    name = "benchmark_serialization",
    srcs = [
        "benchmark_serialization.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/serialization",
        "@benchmark//:benchmark_main",
    ],
)

cc_test(
    name = "benchmark_shm_allocator",
    srcs = [
        "benchmark_shm_allocator.cc",
    ],
    deps = [
        "//neuropod/multiprocess/shm",
        "@benchmark//:benchmark_main",
    ],
)

cc_test(
    name = "test_input_output",
    srcs = [
        "test_input_output.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/serialization",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_serialization",
    srcs = [
        "test_serialization.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/serialization",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_ipc_serialization",
    srcs = [
        "test_ipc_serialization.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess:ipc_control_channel",
        "//neuropod/multiprocess:shm_tensor",
        "//neuropod/multiprocess/serialization",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_ope_wire_format",
    srcs = [
        "test_ope_wire_format.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess/mq",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_ope_heartbeat",
    srcs = [
        "test_ope_heartbeat.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess/mq",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_ope_transferrables",
    srcs = [
        "test_ope_transferrables.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess/mq",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_python_bridge",
    srcs = [
        "test_python_bridge.cc",
    ],
    deps = [
        ":neuropod_test_utils",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/python_bridge",
    ],
)

cc_test(
    name = "gpu_test_python_bridge",
    srcs = [
        "gpu_test_python_bridge.cc",
    ],
    tags = ["gpu"],
    deps = [
        ":neuropod_test_utils",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/python_bridge",
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

cc_library(
    name = "neuropod_test_utils",
    hdrs = ["test_utils.hh"],
    data = [
        "//neuropod/tests/test_data",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":ope_overrides",
        "//neuropod:neuropod_impl",
        "@gtest//:main",
    ],
)

cc_library(
    name = "ope_overrides",
    hdrs = ["ope_overrides.hh"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//neuropod:neuropod_impl",
    ],
)

cc_test(
    name = "test_c_api",
    srcs = [
        "test_c_api.c",
    ],
    copts = ["-std=c11"],
    data = [
        "//neuropod/tests/test_data",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/backends/tensorflow:tensorflow_backend",
        "//neuropod/bindings/c:c_api",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_config_utils",
    srcs = [
        "test_config_utils.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/internal",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_tensor_validation",
    srcs = [
        "test_tensor_validation.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_accessor",
    srcs = [
        "test_accessor.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_shm_allocator",
    srcs = [
        "test_shm_allocator.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess/shm",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_shm_tensor",
    srcs = [
        "test_shm_tensor.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess:shm_tensor",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_ipc_control_channel",
    srcs = [
        "test_ipc_control_channel.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess:ipc_control_channel",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_ipc_message_queue",
    srcs = [
        "test_ipc_message_queue.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess:ipc_control_channel",
        "//neuropod/multiprocess/mq",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_multiprocess_worker",
    srcs = [
        "test_multiprocess_worker.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess:multiprocess_worker",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_multiprocess_allowed_transitions",
    srcs = [
        "test_multiprocess_allowed_transitions.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess:ipc_control_channel",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_multiprocess",
    srcs = [
        "test_multiprocess.cc",
    ],
    data = [
        "//neuropod/backends/python_bridge:libneuropod_pythonbridge_backend.so",
        "//neuropod/backends/tensorflow:libneuropod_tensorflow_backend.so",
        "//neuropod/backends/torchscript:libneuropod_torchscript_backend.so",
    ],
    tags = ["requires_path"],
    deps = [
        ":neuropod_test_utils",
        "//neuropod:neuropod_impl",
        "//neuropod/backends/python_bridge",
        "//neuropod/backends/tensorflow:tensorflow_backend",
        "//neuropod/backends/torchscript:torch_backend",
        "//neuropod/multiprocess",
    ],
)

cc_test(
    name = "test_internal_neuropod_tensor",
    srcs = [
        "test_internal_neuropod_tensor.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/internal",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_conversions_eigen",
    srcs = [
        "test_conversions_eigen.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/conversions:eigen",
        "@eigen3_repo//:eigen3",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_backend_registration",
    srcs = [
        "test_backend_registration.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/internal",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_factories",
    srcs = [
        "test_factories.cc",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_loader",
    srcs = [
        "test_loader.cc",
    ],
    data = [
        "//neuropod/tests/test_data",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/internal",
        "@gtest//:main",
    ],
)
