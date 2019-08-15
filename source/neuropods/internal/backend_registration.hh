//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/internal/config_utils.hh"
#include "neuropods/internal/memory_utils.hh"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuropods
{

class NeuropodBackend;
struct RuntimeOptions;

// A function that takes in a path to a neuropod and returns a pointer to a NeuropodBackend
typedef std::unique_ptr<NeuropodBackend> (*BackendFactoryFunction)(const std::string &           neuropod_path,
                                                                   std::unique_ptr<ModelConfig> &config,
                                                                   const RuntimeOptions &        options);

// A template to create a factory for any backend
// This is used in the macro below
template <typename T>
std::unique_ptr<NeuropodBackend> createNeuropodBackend(const std::string &           neuropod_path,
                                                       std::unique_ptr<ModelConfig> &config,
                                                       const RuntimeOptions &        options)
{
    return stdx::make_unique<T>(neuropod_path, config, options);
}

// Register a backend for a set of specific types
// This is used in the macro below
bool register_backend(const std::string &             name,
                      const std::vector<std::string> &supported_types,
                      BackendFactoryFunction          factory_fn);

// Get a backend factory function for a neuropod type (e.g. "python", "tensorflow", "torchscript")
// `default_backend_overrides` allows users to override the default backend for a given type.
// This is a mapping from a neuropod type to the name of a shared library that supports that type.
// Note: Libraries in this map will only be loaded if a backend for the requested type hasn't already
// been loaded
BackendFactoryFunction get_backend_for_type(
    const std::unordered_map<std::string, std::string> &default_backend_overrides, const std::string &type);

// Get a backend factory function by backend name (e.g. "PythonBridge", "TestNeuropodBackend")
BackendFactoryFunction get_backend_by_name(const std::string &name);

// A macro to easily define a backend
// Example: REGISTER_NEUROPOD_BACKEND(MyPythonBackend, "pytorch", "python")
#define REGISTER_NEUROPOD_BACKEND(CLS, ... /* supported types */) \
    bool is_registered_##CLS = register_backend(#CLS, {__VA_ARGS__}, createNeuropodBackend<CLS>);

} // namespace neuropods
