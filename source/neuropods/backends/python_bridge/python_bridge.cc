//
// Uber, Inc. (c) 2018
//

#include "python_bridge.hh"

#include <dlfcn.h>
#include <exception>
#include <sstream>
#include <stdlib.h>
#include <vector>

#include "neuropods/bindings/python_bindings.hh"

namespace neuropods
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

bool maybe_initialize()
{
    if (Py_IsInitialized())
    {
        return false;
    }

    // If we have a virtualenv, use it
    if (auto venv_path = std::getenv("VIRTUAL_ENV"))
    {
        setenv("PYTHONHOME", venv_path, true);
    }

    // Start the interpreter
    py::initialize_interpreter();

    // TODO: shutdown the interpreter once we know that there are no more python objects left
    // atexit(py::finalize_interpreter);
    return true;
}

// Handle interpreter startup and shutdown
static auto did_initialize = maybe_initialize();

} // namespace

PythonBridge::PythonBridge(const std::string &             neuropod_path,
                           std::unique_ptr<ModelConfig> &  model_config,
                           const std::vector<std::string> &python_path_additions)
{
    // Modify PYTHONPATH
    set_python_path(python_path_additions);

    // Get the neuropod loader
    py::object load_neuropod = py::module::import("neuropods.loader").attr("load_neuropod");

    // Load the neuropod and save a reference to it
    neuropod_ = load_neuropod(neuropod_path);
}

PythonBridge::~PythonBridge() = default;

// Run inference
std::unique_ptr<NeuropodValueMap> PythonBridge::infer(const NeuropodValueMap &inputs)
{
    // Convert to a py::dict
    py::dict model_inputs = to_numpy_dict(const_cast<NeuropodValueMap &>(inputs));

    // Run inference
    py::dict model_outputs = neuropod_.attr("infer")(model_inputs).cast<py::dict>();

    // Get the outputs
    auto outputs = from_numpy_dict(*get_tensor_allocator(), model_outputs);

    // We need a unique pointer
    return stdx::make_unique<NeuropodValueMap>(std::move(outputs));
}

REGISTER_NEUROPOD_BACKEND(PythonBridge, "python", "pytorch")

} // namespace neuropods
