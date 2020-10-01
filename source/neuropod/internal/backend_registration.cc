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

#include "backend_registration.hh"

#include "neuropod/internal/error_utils.hh"

#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <cpp-semver.hpp>
#include <dlfcn.h>

namespace neuropod
{

namespace
{

struct BackendInfo
{
    std::string            version;
    BackendFactoryFunction factory;
};

std::once_flag                                                registrar_initialized;
std::unique_ptr<std::unordered_map<std::string, BackendInfo>> registered_backends_by_type;

void init_registrar_if_needed()
{
    std::call_once(registrar_initialized, []() {
        registered_backends_by_type = stdx::make_unique<std::unordered_map<std::string, BackendInfo>>();

        // Make sure our logging is initialized
        init_logging();
    });
}

// A map from a backend type (e.g. tensorflow, python, torchscript, etc.) and version to
// the name of a shared library that supports that type. These will only be loaded if a backend
// for the requested type hasn't already been loaded.
// Note: these are listed in reverse priority order
// The ordering below means we'd prefer to load a newer, GPU capable version of a framework if one is available
// TODO(vip): Actually name the backend `so` files differently depending on version
const std::vector<BackendLoadSpec> default_backend_for_type = {
    // Torch CPU
    {"torchscript", "1.1.0", "libneuropod_torchscript_backend.so"},
    {"torchscript", "1.2.0", "libneuropod_torchscript_backend.so"},
    {"torchscript", "1.3.0", "libneuropod_torchscript_backend.so"},
    {"torchscript", "1.4.0", "libneuropod_torchscript_backend.so"},

    // Torch GPU
    {"torchscript", "1.1.0", "libneuropod_torchscript_backend.so"},
    {"torchscript", "1.2.0", "libneuropod_torchscript_backend.so"},
    {"torchscript", "1.3.0", "libneuropod_torchscript_backend.so"},
    {"torchscript", "1.4.0", "libneuropod_torchscript_backend.so"},

    // TF CPU
    {"tensorflow", "1.12.0", "libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.13.1", "libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.14.0", "libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.15.0", "libneuropod_tensorflow_backend.so"},

    // TF GPU
    {"tensorflow", "1.12.0", "libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.13.1", "libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.14.0", "libneuropod_tensorflow_backend.so"},
    {"tensorflow", "1.15.0", "libneuropod_tensorflow_backend.so"},

    // Python
    {"python", "27", "libneuropod_pythonbridge_backend.so"},
    {"python", "35", "libneuropod_pythonbridge_backend.so"},
    {"python", "36", "libneuropod_pythonbridge_backend.so"},
    {"python", "37", "libneuropod_pythonbridge_backend.so"},
    {"python", "38", "libneuropod_pythonbridge_backend.so"},
};

bool load_default_backend(const std::vector<BackendLoadSpec> &backends,
                          const std::string &                 type,
                          const std::string &                 target_version_range)
{
    // Reverse priority order
    for (auto it = backends.rbegin(); it != backends.rend(); it++)
    {
        const auto &backend = *it;
        if (backend.type != type)
        {
            // Type doesn't match
            continue;
        }

        if (!semver::satisfies(backend.version, target_version_range))
        {
            // Target version range isn't satisifed
            continue;
        }

        // Try to dlopen it
        const auto &sopath = backend.path;
        if (dlopen(sopath.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr)
        {
            const auto err = dlerror();
            if (err == nullptr)
            {
                SPDLOG_TRACE("Loading the default backend for type '{}' failed, but no error message was avaliable",
                             type);
            }
            else
            {
                SPDLOG_TRACE("Loading the default backend for type '{}' failed. Error from dlopen: {}", type, err);
            }

            return false;
        }

        // Successfully loaded the backend
        SPDLOG_TRACE("Successfully loaded default backend '{}'", sopath);
        return true;
    }

    return false;
}

BackendFactoryFunction find_registered_backend(const std::string &type, const std::string &target_version_range)
{
    auto it = registered_backends_by_type->find(type);
    if (it != registered_backends_by_type->end())
    {
        // Check if it matches the version range
        if (semver::satisfies(it->second.version, target_version_range))
        {
            return it->second.factory;
        }
        else
        {
            SPDLOG_TRACE("Version '{}' for backend '{}' does not satisfy the requested version range '{}'",
                         it->second.version,
                         type,
                         target_version_range);
        }
    }
    else
    {
        SPDLOG_TRACE("Unable to find backend for type '{}' in backend registry which contains '{}' elements.",
                     type,
                     registered_backends_by_type->size());
    }

    return nullptr;
}

} // namespace

bool BackendLoadSpec::operator==(const BackendLoadSpec &other) const
{
    return type == other.type && version == other.version && path == other.path;
}

bool register_backend(const std::string &    name,
                      const std::string &    type,
                      const std::string &    version,
                      BackendFactoryFunction factory_fn)
{
    init_registrar_if_needed();

    SPDLOG_DEBUG("Registering backend {} with type {} and version {}", name, type, version);

    BackendInfo info;
    info.version = version;
    info.factory = std::move(factory_fn);

    if (!semver::valid(version))
    {
        NEUROPOD_ERROR("Tried registering backend {} with type {} and version {}, but the specified version is not a "
                       "valid semver version. See https://semver.org/ for more details.",
                       name,
                       type,
                       version);
    }

    // Check to see if we've already loaded a backend for `type`
    // We can only load one backend version per type into the same process
    // (i.e. if we load TF 1.12.0, we can't load another version of TF into this process)
    // Using OPE overcomes this problem
    if (registered_backends_by_type->find(type) != registered_backends_by_type->end())
    {
        NEUROPOD_ERROR(
            "Attempted to register a backend for type '{}', but one was already loaded. If you are trying "
            "to use multiple versions of the same framework, you must use OPE. See the docs at https://neuropod.ai",
            type);
    }

    registered_backends_by_type->insert(std::make_pair(type, info));

    return true;
}

BackendFactoryFunction get_backend_for_type(const std::vector<BackendLoadSpec> &default_backend_overrides,
                                            const std::string &                 type,
                                            const std::string &                 target_version_range)
{
    init_registrar_if_needed();

    {
        // Attempt to find a registered backend that matches
        auto retval = find_registered_backend(type, target_version_range);
        if (retval != nullptr)
        {
            return retval;
        }
    }

    // Check to see if we've already loaded a backend for `type`
    // We can only load one backend version per type into the same process
    // (i.e. if we load TF 1.12.0, we can't load another version of TF into this process)
    // Using OPE overcomes this problem
    auto it = registered_backends_by_type->find(type);
    if (it != registered_backends_by_type->end())
    {
        NEUROPOD_ERROR(
            "Tried to get a backend for type '{}' and version range '{}' but failed. A backend for type "
            "'{}' was already registered, but it supports version '{}'. If you are trying "
            "to use multiple versions of the same framework, you must use OPE. See the docs at https://neuropod.ai",
            type,
            target_version_range,
            type,
            it->second.version);
    }

    // Try loading using the user provided overrides
    bool load_success = load_default_backend(default_backend_overrides, type, target_version_range);
    if (!load_success)
    {
        // If that didn't work, try loading using the default map
        load_success = load_default_backend(default_backend_for_type, type, target_version_range);
    }

    if (load_success)
    {
        // We loaded a library so we can check the registered backends again
        auto retval = find_registered_backend(type, target_version_range);
        if (retval != nullptr)
        {
            return retval;
        }
    }

    // Don't have anything that matches
    NEUROPOD_ERROR("The model being loaded requires a Neuropod backend for type '{}' and version range '{}'. However, "
                   "a backend satisfying these requirements was not found. See the installation instructions "
                   "at https://neuropod.ai to install a backend. Retry with log level TRACE for more information.",
                   type,
                   target_version_range);
}

} // namespace neuropod
