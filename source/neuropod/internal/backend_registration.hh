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

struct BackendLoadSpec
{
    // A neuropod platform (e.g. "python", "tensorflow", "torchscript")
    std::string type;

    // The version of the platform (e.g. "1.12.0")
    std::string version;

    // The name or path of a shared library that supports the above platform and version
    // (e.g. "libneuropod_tensorflow_backend.so")
    std::string path;

    bool operator==(const BackendLoadSpec &other) const;
};

// Get a backend factory function for a neuropod type (e.g. "python", "tensorflow", "torchscript")
// `default_backend_overrides` allows users to override the default backend for a given type.
// This is a mapping from a neuropod type to the name of a shared library that supports that type.
// Note: Libraries in this map will only be loaded if a backend for the requested type hasn't already
// been loaded
BackendFactoryFunction get_backend_for_type(const std::vector<BackendLoadSpec> &default_backend_overrides,
                                            const std::string &                 type,
                                            const std::string &                 target_version_range = "*");

// A macro to easily define a backend
// Example: REGISTER_NEUROPOD_BACKEND(SomeTensorflowBackend, "tensorflow", "1.13.1")
#define REGISTER_NEUROPOD_BACKEND(CLS, type, version)                                             \
    namespace                                                                                     \
    {                                                                                             \
    bool is_registered_##CLS = register_backend(#CLS, type, version, createNeuropodBackend<CLS>); \
    }

// Utility macros
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

} // namespace neuropod
