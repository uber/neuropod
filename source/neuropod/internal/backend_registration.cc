//
// Uber, Inc. (c) 2018
//

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

std::once_flag                                                     registrar_initialized;
std::unique_ptr<std::unordered_multimap<std::string, BackendInfo>> registered_backends_by_type;

void init_registrar_if_needed()
{
    std::call_once(registrar_initialized, []() {
        registered_backends_by_type = stdx::make_unique<std::unordered_multimap<std::string, BackendInfo>>();
    });
}

// A map from a backend type (e.g. tensorflow, python, torchscript, etc.) to the name
// of a shared library that supports that type. These will only be loaded if a backend
// for the requested type hasn't already been loaded
const std::unordered_map<std::string, std::string> default_backend_for_type = {
    {"tensorflow", "libneuropod_tensorflow_backend.so"},
    {"python", "libneuropod_pythonbridge_backend.so"},
    {"pytorch", "libneuropod_pythonbridge_backend.so"},
    {"torchscript", "libneuropod_torchscript_backend.so"},
};

void load_default_backend(const std::unordered_map<std::string, std::string> &default_backend_overrides,
                          const std::string &                                 type)
{
    auto backend_it = default_backend_overrides.find(type);
    if (backend_it == default_backend_overrides.end())
    {
        // This type is not in the overrides so check the default mapping
        backend_it = default_backend_for_type.find(type);
    }

    if (backend_it == default_backend_for_type.end())
    {
        NEUROPOD_ERROR("Default Neuropod backend not found for type '{}'! "
                       "Make sure that you load a Neuropod backend that can support '{}'",
                       type,
                       type);
    }
    else
    {
        if (dlopen(backend_it->second.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr)
        {
            NEUROPOD_ERROR("Loading the default backend for type '{}' failed. Error from dlopen: {}", type, dlerror());
        }
    }
}

} // namespace

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

    registered_backends_by_type->insert(std::make_pair(type, info));

    return true;
}

BackendFactoryFunction get_backend_for_type(
    const std::unordered_map<std::string, std::string> &default_backend_overrides,
    const std::string &                                 type,
    const std::string &                                 target_version_range)
{
    init_registrar_if_needed();

    if (registered_backends_by_type->find(type) == registered_backends_by_type->end())
    {
        // We don't have a backend loaded for the requested type
        // Try loading the default one
        load_default_backend(default_backend_overrides, type);
    }

    auto range = registered_backends_by_type->equal_range(type);
    for (auto it = range.first; it != range.second; it++)
    {
        // Check if it matches the version range
        if (semver::satisfies(it->second.version, target_version_range))
        {
            return it->second.factory;
        }
    }

    // If we get here, that means that we don't have any matching backends
    NEUROPOD_ERROR(
        "Neuropod backend not found for type '{}' and version range '{}'! Default backend did not match either.",
        type,
        target_version_range);
}

} // namespace neuropod
