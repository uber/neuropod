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

#include "python_bridge.hh"

#include "neuropod/bindings/python_bindings.hh"
#include "neuropod/internal/error_utils.hh"

#include <cstdlib>
#include <exception>
#include <sstream>
#include <vector>

namespace neuropod
{

namespace
{

void set_python_path(const std::vector<std::string> &paths_to_add)
{
    std::stringstream python_path;

    for (const auto &dir : paths_to_add)
    {
        python_path << dir << ":";
    }

    if (const char *existing = std::getenv("PYTHONPATH"))
    {
        python_path << existing;
    }

    // Overwrite the existing PYTHONPATH with the new one
    setenv("PYTHONPATH", python_path.str().c_str(), 1);
}

} // namespace

PythonBridge::PythonBridge(const std::string &             neuropod_path,
                           const RuntimeOptions &          options,
                           const std::vector<std::string> &python_path_additions)
    : NeuropodBackendWithDefaultAllocator<GenericNeuropodTensor>(neuropod_path, options),
      py_interpreter_handle_(get_interpreter_handle())
{
    // Modify PYTHONPATH
    set_python_path(python_path_additions);

    if (options.load_model_at_construction)
    {
        load_model();
    }
}

void PythonBridge::load_model_internal()
{
    // Acquire the GIL
    py::gil_scoped_acquire gil;

    // Get the python neuropod loader
    py::object load_neuropod = py::module::import("neuropod.backends.python.executor").attr("PythonNeuropodExecutor");

    // Converts from unicode to ascii for python 3 string arrays
    maybe_convert_bindings_types_ = stdx::make_unique<py::object>(
        py::module::import("neuropod.utils.dtype_utils").attr("maybe_convert_bindings_types"));

    // Make sure that the model is local
    // Note: we could also delegate this to the python implementation
    const auto local_path = loader_->ensure_local();

    // Load the neuropod and save a reference to it
    neuropod_ = stdx::make_unique<py::object>(load_neuropod(local_path));
}

PythonBridge::~PythonBridge()
{
    // Acquire the GIL
    py::gil_scoped_acquire gil;

    // Delete the stored objects
    maybe_convert_bindings_types_.reset();
    neuropod_.reset();
}

// Run inference
std::unique_ptr<NeuropodValueMap> PythonBridge::infer_internal(const NeuropodValueMap &inputs)
{
    // Acquire the GIL
    py::gil_scoped_acquire gil;

    // Convert to a py::dict
    py::dict model_inputs = to_numpy_dict(const_cast<NeuropodValueMap &>(inputs));

    // Run inference
    auto model_outputs_raw = neuropod_->attr("infer")(model_inputs).cast<py::dict>();

    // Postprocess for python 3
    auto model_outputs = (*maybe_convert_bindings_types_)(model_outputs_raw).cast<py::dict>();

    // Get the outputs
    auto outputs = from_numpy_dict(*get_tensor_allocator(), model_outputs);

    // We need a unique pointer
    return stdx::make_unique<NeuropodValueMap>(std::move(outputs));
}

REGISTER_NEUROPOD_BACKEND(PythonBridge, "python", PY_VERSION)

} // namespace neuropod
