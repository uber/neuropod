/* Copyright (c) 2020 UATC, LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include "neuropod/neuropod.hh"

#include <string>
#include <unordered_map>

#include <unistd.h>

namespace detail
{

std::string get_cwd()
{
    char buf[FILENAME_MAX];
    getcwd(buf, FILENAME_MAX);

    return buf;
}

const auto CWD = get_cwd();

// An override for the default backend locations. Used by tests
const std::vector<neuropod::BackendLoadSpec> ope_backend_location_overrides = {
    // Torch CPU
    {"torchscript", "1.1.0", CWD + "/neuropod/backends/torchscript/libneuropod_torchscript_backend.so"},
    {"torchscript", "1.2.0", CWD + "/neuropod/backends/torchscript/libneuropod_torchscript_backend.so"},
    {"torchscript", "1.3.0", CWD + "/neuropod/backends/torchscript/libneuropod_torchscript_backend.so"},
    {"torchscript", "1.4.0", CWD + "/neuropod/backends/torchscript/libneuropod_torchscript_backend.so"},

    // Torch GPU
    {"torchscript", "1.1.0", CWD + "/neuropod/backends/torchscript/libneuropod_torchscript_backend.so"},
    {"torchscript", "1.2.0", CWD + "/neuropod/backends/torchscript/libneuropod_torchscript_backend.so"},
    {"torchscript", "1.3.0", CWD + "/neuropod/backends/torchscript/libneuropod_torchscript_backend.so"},
    {"torchscript", "1.4.0", CWD + "/neuropod/backends/torchscript/libneuropod_torchscript_backend.so"},

    // TF CPU
    {"tensorflow", "1.12.0", CWD + "/neuropod/backends/tensorflow/libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.13.1", CWD + "/neuropod/backends/tensorflow/libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.14.0", CWD + "/neuropod/backends/tensorflow/libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.15.0", CWD + "/neuropod/backends/tensorflow/libneuropod_tensorflow_backend.so"},

    // TF GPU
    {"tensorflow", "1.12.0", CWD + "/neuropod/backends/tensorflow/libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.13.1", CWD + "/neuropod/backends/tensorflow/libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.14.0", CWD + "/neuropod/backends/tensorflow/libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.15.0", CWD + "/neuropod/backends/tensorflow/libneuropod_tensorflow_backend.so"},

    // Python
    {"python", "27", CWD + "/neuropod/backends/python_bridge/libneuropod_pythonbridge_backend.so"},
    {"python", "35", CWD + "/neuropod/backends/python_bridge/libneuropod_pythonbridge_backend.so"},
    {"python", "36", CWD + "/neuropod/backends/python_bridge/libneuropod_pythonbridge_backend.so"},
    {"python", "37", CWD + "/neuropod/backends/python_bridge/libneuropod_pythonbridge_backend.so"},
    {"python", "38", CWD + "/neuropod/backends/python_bridge/libneuropod_pythonbridge_backend.so"},
};

} // namespace detail
