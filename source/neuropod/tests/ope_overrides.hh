//
// Uber, Inc. (c) 2020
//

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
