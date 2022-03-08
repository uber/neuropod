/* Copyright (c) 2020 The Neuropod Authors

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

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/core/generic_tensor.hh"

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
class PythonBridge : public NeuropodBackendWithDefaultAllocator<GenericNeuropodTensor>
{
private:
    std::unique_ptr<py::object> neuropod_;

public:
    PythonBridge(const std::string &             neuropod_path,
                 const RuntimeOptions &          options,
                 const std::vector<std::string> &python_path_additions = get_default_python_path());

    ~PythonBridge();

protected:
    // Run inference
    std::unique_ptr<NeuropodValueMap> infer_internal(const NeuropodValueMap &inputs);

    // A method that loads the underlying model
    void load_model_internal();
};

} // namespace neuropod
