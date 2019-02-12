//
// Uber, Inc. (c) 2018
//

#pragma once

#include <boost/python.hpp>
#include <string>
#include <vector>

#include "neuropods/backends/neuropod_backend.hh"

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
class PythonBridge : public NeuropodBackend
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

    // Allocate a tensor of a specific type
    std::unique_ptr<NeuropodTensor> allocate_tensor(const std::string &         node_name,
                                                    const std::vector<int64_t> &input_dims,
                                                    TensorType                  tensor_type);

    // Run inference
    std::unique_ptr<TensorStore> infer(const TensorStore &inputs);
};

} // namespace neuropods
