//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/backends/test_backend/test_neuropod_tensor.hh"

#include <pybind11/embed.h>

#include <string>
#include <vector>

namespace neuropod
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
    std::unique_ptr<py::object> neuropod_;
    std::unique_ptr<py::object> maybe_convert_bindings_types_;

public:
    PythonBridge(const std::string &             neuropod_path,
                 const RuntimeOptions &          options,
                 const std::vector<std::string> &python_path_additions = get_default_python_path());

    ~PythonBridge();

protected:
    // Run inference
    std::unique_ptr<NeuropodValueMap> infer_internal(const NeuropodValueMap &inputs);
};

} // namespace neuropod
