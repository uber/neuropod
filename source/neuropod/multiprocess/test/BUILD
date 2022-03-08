# Copyright (c) 2021 The Neuropod Authors
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
    name = "benchmark_multiprocess",
    srcs = [
        "benchmark_multiprocess.cc",
    ],
    data = [
        "//neuropod/tests/test_data",
    ],
    tags = [
        # Don't do trace logging in CI for this benchmark
        "no_trace_logging",
        "requires_framework_tensorflow",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/backends/tensorflow:tensorflow_backend",
        "//neuropod/multiprocess",
        "@benchmark//:benchmark_main",
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
    name = "test_ope_multiple_instances",
    srcs = [
        "test_ope_multiple_instances.cc",
    ],
    tags = [
        "no_trace_logging",
        "requires_framework_python",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/multiprocess",
        "//neuropod/tests:neuropod_test_utils",
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
