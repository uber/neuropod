# Copyright (c) 2020 The Neuropod Authors
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

cc_library(
    name = "shm_tensor",
    srcs = [
        "shm_tensor.cc",
        "shm_tensor.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal",
        "//neuropod/multiprocess/serialization",
        "//neuropod/multiprocess/shm",
    ],
)

cc_library(
    name = "ipc_control_channel",
    srcs = [
        "control_messages.cc",
        "ipc_control_channel.cc",
    ],
    hdrs = [
        "control_messages.hh",
        "ipc_control_channel.hh",
        "ope_load_config.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":shm_tensor",
        "//neuropod/multiprocess/mq",
    ],
)

cc_library(
    name = "multiprocess_worker",
    srcs = [
        "multiprocess_worker.cc",
        "tensor_utils.hh",
    ],
    hdrs = [
        "multiprocess_worker.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":ipc_control_channel",
        "//neuropod:neuropod_hdrs",
        "//neuropod/internal",
    ],
)

cc_binary(
    name = "neuropod_multiprocess_worker",
    srcs = [
        "multiprocess_worker_main.cc",
        "//neuropod:libneuropod.so",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = True,
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":multiprocess_worker",
    ],
)

cc_library(
    name = "multiprocess",
    hdrs = [
        "multiprocess.hh",
    ],
    data = [
        ":neuropod_multiprocess_worker",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        "//neuropod:neuropod_hdrs",
    ],
)

cc_library(
    name = "impl",
    srcs = [
        "multiprocess.cc",
    ],
    hdrs = [
        "multiprocess.hh",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
    deps = [
        ":ipc_control_channel",
        "//neuropod:neuropod_hdrs",
        "//neuropod/backends:neuropod_backend",
        "//neuropod/internal",
        "@boost_repo//:boost",
    ],
)
