//
// Uber, Inc. (c) 2018
//

#include "python_bridge.hh"

#include <dlfcn.h>
#include <exception>
#include <sstream>
#include <stdlib.h>
#include <vector>

#include "neuropods/backends/python_bridge/type_utils.hh"
#include "neuropods/internal/tensor_store.hh"

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

// We can't shutdown python in the destructor of `PythonBridge`
// because there still might be NumpyNeuropodTensors floating around
// Instead, we shutdown at process termination
void shutdown_python()
{
    // Boost Python requires us not to call Py_Finalize
    // https://www.boost.org/doc/libs/1_62_0/libs/python/doc/html/tutorial/tutorial/embedding.html
    // Py_Finalize();
}

// Make sure we only shutdown python once
static bool did_register_shutdown = false;

void maybe_register_python_shutdown()
{
    if (!did_register_shutdown)
    {
        atexit(shutdown_python);
        did_register_shutdown = true;
    }
}

} // namespace

PythonBridge::PythonBridge(const std::string &             neuropod_path,
                           std::unique_ptr<ModelConfig>    model_config,
                           const std::vector<std::string> &python_path_additions)
     : NeuropodBackend(std::move(model_config))
{
    try
    {
        // Utilize the virtualenv if one is present
        if (auto venv_path = std::getenv("VIRTUAL_ENV"))
        {
            setenv("PYTHONHOME", venv_path, true);
        }

        // Register python shutdown
        maybe_register_python_shutdown();

        // Modify PYTHONPATH
        set_python_path(python_path_additions);

        // Workaround for this issue:
        // https://stackoverflow.com/questions/11842920/undefined-symbol-pyexc-importerror-when-embedding-python-in-c
        dlopen("libpython2.7.so", RTLD_LAZY | RTLD_GLOBAL);

        // Initialize the embedded python interpreter
        Py_Initialize();

        // Initial setup
        main_module_    = py::import("__main__");
        main_namespace_ = main_module_.attr("__dict__");

        // Create a local python variable with the neuropod model path
        py::dict locals;
        locals["neuropod_path"] = neuropod_path;

        // Load the neuropod
        py::exec("from neuropods.loader import load_neuropod\n"
                 "neuropod = load_neuropod(neuropod_path)\n",
                 main_namespace_,
                 locals);

        // Save a reference to the loaded neuropod
        neuropod_ = locals["neuropod"];
    }
    catch (py::error_already_set)
    {
        PyErr_Print();
        throw;
    }
}

PythonBridge::~PythonBridge() = default;

// Run inference
std::unique_ptr<TensorStore> PythonBridge::infer(const std::unordered_set<std::shared_ptr<NeuropodTensor>> &inputs)
{
    try
    {
        // Populate a dict mapping input names to values
        py::dict model_inputs;
        for (const std::shared_ptr<NeuropodTensor> &tensor : inputs)
        {
            model_inputs[tensor->get_name()]
                = std::dynamic_pointer_cast<NativeDataContainer<py::object>>(tensor)->get_native_data();
        }

        // Create local python variables with the loaded neuropod and the input dict
        // we just created
        py::dict locals;
        locals["neuropod"]     = neuropod_;
        locals["model_inputs"] = model_inputs;

        // Run inference
        py::exec("model_outputs = neuropod.infer(model_inputs)\n", main_namespace_, locals);

        // Get the output
        py::dict model_outputs = py::extract<py::dict>(locals["model_outputs"]);

        // Convert from numpy to `NeuropodTensor`s
        std::unique_ptr<TensorStore> to_return = stdx::make_unique<TensorStore>();
        py::list                     out_keys  = model_outputs.keys();
        for (int i = 0; i < py::len(out_keys); i++)
        {
            const char *      key_c_str = py::extract<const char *>(out_keys[i]);
            const int         key_len   = py::extract<int>(out_keys[i].attr("__len__")());
            const std::string key       = std::string(key_c_str, key_len);

            PyArrayObject *nparray     = get_nparray_from_obj(model_outputs[key]);
            TensorType     tensor_type = get_neuropod_type_from_numpy_type(PyArray_TYPE(nparray));
            to_return->tensors.emplace_back(make_tensor<NumpyNeuropodTensor>(tensor_type, key, nparray));
        }

        return to_return;
    }
    catch (py::error_already_set)
    {
        PyErr_Print();
        throw;
    }
}

REGISTER_NEUROPOD_BACKEND(PythonBridge, "python", "pytorch")

} // namespace neuropods
