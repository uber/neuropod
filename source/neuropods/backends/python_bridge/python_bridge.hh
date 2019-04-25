//
// Uber, Inc. (c) 2018
//

#pragma once

#include <boost/python.hpp>
#include <string>
#include <vector>

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/backends/python_bridge/numpy_neuropod_tensor.hh"

namespace neuropods
{

namespace py = boost::python;

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
class PythonBridge : public NeuropodBackendWithDefaultAllocator<NumpyNeuropodTensor>
{
private:
    py::object main_module_;
    py::object main_namespace_;

    py::object neuropod_;

public:
    PythonBridge(const std::string &             neuropod_path,
                 std::unique_ptr<ModelConfig> &  model_config,
                 const std::vector<std::string> &python_path_additions = get_default_python_path());

    ~PythonBridge();

    // Run inference
    std::unique_ptr<ValueMap> infer(const ValueSet &inputs);
};

} // namespace neuropods
