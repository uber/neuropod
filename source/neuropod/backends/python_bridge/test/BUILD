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
    name = "test_python_bridge",
    srcs = [
        "test_python_bridge.cc",
    ],
    # Needed because we want to expose symbols and we're
    # dynamically loading the python bridge
    linkopts = ["-rdynamic"],
    tags = ["requires_framework_python"],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/tests:neuropod_test_utils",
    ],
)

cc_test(
    name = "gpu_test_python_bridge",
    srcs = [
        "gpu_test_python_bridge.cc",
    ],
    # Needed because we want to expose symbols and we're
    # dynamically loading the python bridge
    linkopts = ["-rdynamic"],
    tags = [
        "gpu",
        "requires_framework_python",
    ],
    deps = [
        "//neuropod:neuropod_impl",
        "//neuropod/tests:neuropod_test_utils",
    ],
)
