//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropod/internal/config_utils.hh"
#include "neuropod/internal/memory_utils.hh"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuropod
{

class NeuropodBackend;
struct RuntimeOptions;

// A function that takes in a path to a neuropod and returns a pointer to a NeuropodBackend
typedef std::unique_ptr<NeuropodBackend> (*BackendFactoryFunction)(const std::string &   neuropod_path,
                                                                   const RuntimeOptions &options);

// A template to create a factory for any backend
// This is used in the macro below
template <typename T>
std::unique_ptr<NeuropodBackend> createNeuropodBackend(const std::string &neuropod_path, const RuntimeOptions &options)
{
    return stdx::make_unique<T>(neuropod_path, options);
}

// Register a backend for a set of specific types
// This is used in the macro below
bool register_backend(const std::string &    name,
                      const std::string &    type,
                      const std::string &    version,
                      BackendFactoryFunction factory_fn);

// Get a backend factory function for a neuropod type (e.g. "python", "tensorflow", "torchscript")
// `default_backend_overrides` allows users to override the default backend for a given type.
// This is a mapping from a neuropod type to the name of a shared library that supports that type.
// Note: Libraries in this map will only be loaded if a backend for the requested type hasn't already
// been loaded
BackendFactoryFunction get_backend_for_type(
    const std::unordered_map<std::string, std::string> &default_backend_overrides,
    const std::string &                                 type,
    const std::string &                                 target_version_range = "*");

// A macro to easily define a backend
// Example: REGISTER_NEUROPOD_BACKEND(SomeTensorflowBackend, "tensorflow", "1.13.1")
#define REGISTER_NEUROPOD_BACKEND(CLS, type, version) \
    bool is_registered_##CLS = register_backend(#CLS, type, version, createNeuropodBackend<CLS>);

// Utility macros
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

} // namespace neuropod
