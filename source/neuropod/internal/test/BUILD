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
