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

#include "python_bridge.hh"

#include "neuropod/bindings/python_bindings.hh"
#include "neuropod/internal/error_utils.hh"

#include <ghc/filesystem.hpp>

#include <cstdlib>
#include <exception>
#include <sstream>
#include <vector>

#include <dlfcn.h>

namespace neuropod
{

namespace
{

namespace fs = ghc::filesystem;

// Returns the path of the so that contains this function
// This is so we can set PYTHONHOME correctly
const char *get_current_so_path()
{
    Dl_info info;
    dladdr(reinterpret_cast<const void *>(get_current_so_path), &info);
    return info.dli_fname;
}

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

// Initialize python if necessary and make sure we don't lock the GIL
std::unique_ptr<py::gil_scoped_release> maybe_initialize()
{
    // Make sure our logging is initialized
    init_logging();

    if (Py_IsInitialized()) // NOLINT(readability-implicit-bool-conversion)
    {
        return nullptr;
    }

#ifndef __APPLE__
// This binary is already linked against `libpython`; the dlopen just
// promotes it to RTLD_GLOBAL.
#define PYTHON_LIB_NAME "libpython" STR(PYTHON_VERSION) ".so.1.0"
#define PYTHON_LIB_M_NAME "libpython" STR(PYTHON_VERSION) "m.so.1.0"
    void *libpython = dlopen(PYTHON_LIB_NAME, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);

    if (libpython == nullptr)
    {
        libpython = dlopen(PYTHON_LIB_M_NAME, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    }

    if (libpython == nullptr)
    {
        const auto err = dlerror();
        if (err == nullptr)
        {
            NEUROPOD_ERROR("Failed to promote libpython to RTLD_GLOBAL; this likely means the neuropod backend library "
                           "was not built correctly");
        }
        else
        {
            NEUROPOD_ERROR("Failed to promote libpython to RTLD_GLOBAL. Error from dlopen: {}", err);
        }
    }
#endif

    // Get the current backend directory
    auto sopath = get_current_so_path();
    if (sopath == nullptr)
    {
        NEUROPOD_ERROR("Error getting path of current shared object. Cannot load python.");
    }

    const auto sodir = fs::absolute(sopath).parent_path();

    // Get the path for pythonhone
#ifdef __APPLE__
    const auto pythonhome = (sodir / "Python.framework/Versions/Current").string();
#else
    const auto pythonhome =
        (sodir / ("opt/python" + std::to_string(PY_MAJOR_VERSION) + "." + std::to_string(PY_MINOR_VERSION))).string();
#endif

    if (std::getenv("NEUROPOD_DISABLE_PYTHON_ISOLATION") == nullptr)
    {
        // Isolate from the environment, set PYTOHNHOME to the packaged python environment
        SPDLOG_TRACE("Setting PYTHONHOME to isolated environment at {}", pythonhome);
        setenv("PYTHONHOME", pythonhome.c_str(), true); // NOLINT(readability-implicit-bool-conversion)
    }
    else if (std::getenv("PYTHONHOME") == nullptr)
    {
        // We're not being asked to isolate from the environment and we don't have pythonhome already set

        // Check if we have a virtualenv
        if (auto venv_path = std::getenv("VIRTUAL_ENV"))
        {
            setenv("PYTHONHOME", venv_path, true); // NOLINT(readability-implicit-bool-conversion)
        }
    }

    // Start the interpreter
    py::initialize_interpreter();

    // pybind11 adds the current working dir to the path and we don't want that
    py::exec(R"(
        import sys
        if sys.path[-1] == ".":
            sys.path.pop()
    )");

    // Add the bootstrap library to the pythonpath
    py::module::import("sys").attr("path").cast<py::list>().append((sodir / "bootstrap").string());

    // Set the executable path
    py::module::import("sys").attr("executable") =
        (fs::path(pythonhome) /
         ("bin/python" + std::to_string(PY_MAJOR_VERSION) + "." + std::to_string(PY_MINOR_VERSION)))
            .string();

    // For code coverage
    if (std::getenv("COVERAGE_PROCESS_START") != nullptr)
    {
        // Get the coverage dependency
        py::module::import("_neuropod_native_bootstrap.pip_utils").attr("_load_deps_internal")("coverage==5.3");

        // Start coverage collection
        py::module::import("coverage").attr("process_startup")();
    }

    // TODO: shutdown the interpreter once we know that there are no more python objects left
    // atexit(py::finalize_interpreter);
    return stdx::make_unique<py::gil_scoped_release>();
}

// Handle interpreter startup and shutdown
// If we initialized the interpreter, make sure we don't have a lock on the GIL by storing a
// py::gil_scoped_release
auto gil_release = maybe_initialize();

} // namespace

PythonBridge::PythonBridge(const std::string &             neuropod_path,
                           const RuntimeOptions &          options,
                           const std::vector<std::string> &python_path_additions)
    : NeuropodBackendWithDefaultAllocator<GenericNeuropodTensor>(neuropod_path, options)
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

    // Bootstrap any deps needed by the Neuropod python library
    py::module::import("_neuropod_native_bootstrap.pip_utils").attr("bootstrap_requirements")();

    // Get the python neuropod loader
    py::object load_neuropod = py::module::import("_neuropod_native_bootstrap.executor").attr("NativePythonExecutor");

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
    neuropod_.reset();

    // Write coverage info if necessary
    // This is necessary because coveragepy depends on atexit. atexit is only called
    // when Py_Finalize is called. Unfortunately, calling Py_Finalize with embedded
    // interpreters is not straightforward.
    // See https://github.com/uber/neuropod/pull/448#issuecomment-704095542 for more details.
    auto sys_modules = py::module::import("sys").attr("modules").cast<py::dict>();
    if (sys_modules.contains("coverage"))
    {
        py::object process_startup = sys_modules["coverage"].attr("process_startup");
        if (py::hasattr(process_startup, "coverage"))
        {
            // Call the atexit handler
            process_startup.attr("coverage").attr("_atexit")();
        }
    }
}

// Run inference
std::unique_ptr<NeuropodValueMap> PythonBridge::infer_internal(const NeuropodValueMap &inputs)
{
    // Acquire the GIL
    py::gil_scoped_acquire gil;

    // Convert to a py::dict
    py::dict model_inputs = to_numpy_dict(const_cast<NeuropodValueMap &>(inputs));

    // Run inference
    auto model_outputs = neuropod_->attr("forward")(model_inputs).cast<py::dict>();

    // Get the outputs
    auto outputs = from_numpy_dict(*get_tensor_allocator(), model_outputs);

    // We need a unique pointer
    return stdx::make_unique<NeuropodValueMap>(std::move(outputs));
}

REGISTER_NEUROPOD_BACKEND(PythonBridge, "python", PY_VERSION)

} // namespace neuropod
