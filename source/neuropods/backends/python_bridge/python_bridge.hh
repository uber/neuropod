//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/backends/test_backend/test_neuropod_tensor.hh"

#include <pybind11/embed.h>

#include <string>
#include <vector>

namespace neuropods
{

namespace py = pybind11;

namespace
{

// Get the default PYTHONPATH additions
std::vector<std::string> get_default_python_path()
{
    return {};
}

} // namespace

// This backend starts an embedded python interpreter and is used
// to execute neuropods that contain python code. This includes
// models from PyTorch < 1.0 and PyTorch models that don't use TorchScript
class PythonBridge : public NeuropodBackendWithDefaultAllocator<TestNeuropodTensor>
{
private:
    py::object neuropod_;
    py::object maybe_convert_bindings_types_;

public:
    PythonBridge(const std::string &             neuropod_path,
                 std::unique_ptr<ModelConfig> &  model_config,
                 const std::vector<std::string> &python_path_additions = get_default_python_path());

    ~PythonBridge();

    // Run inference
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &inputs);
};

} // namespace neuropods
