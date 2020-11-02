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
#include "neuropod/version.hh"

#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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

std::vector<BackendLoadSpec> get_default_backend_map()
{
    std::vector<BackendLoadSpec> out;

    // A structure to store some basic info about frameworks we support
    struct FrameworkInfo
    {
        std::string              type;
        std::string              soname;
        bool                     has_gpu_version;
        std::vector<std::string> versions;
    };

    // Information about frameworks that we use to generate a list of `BackendLoadSpec`
    const std::vector<FrameworkInfo> frameworks = {
        {"torchscript",
         "libneuropod_torchscript_backend.so",
         true,
         {"1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"}},
        {"tensorflow", "libneuropod_tensorflow_backend.so", true, {"1.12.0", "1.13.1", "1.14.0", "1.15.0", "2.2.0"}},
        {"python", "libneuropod_pythonbridge_backend.so", false, {"2.7", "3.5", "3.6", "3.7", "3.8"}}};

    // Base directory for Neuropod backends
    std::string neuropod_base_dir = "/usr/local/lib/neuropod";
    if (auto base_dir = std::getenv("NEUROPOD_BASE_DIR"))
    {
        neuropod_base_dir = base_dir;
    }

    // Because the returned vector is in reverse priority order,
    // the ordering below means we'd prefer to load a newer, GPU capable version of a framework if one is available.
    // It also prioritizes non-absoltute so paths (i.e. controlled by LD_LIBRARY_PATH) over absolute ones in
    // `/usr/local/lib/neuropod`. This is so we don't break existing behavior.
    for (const auto &is_absolute_path : {false, true})
    {
        for (const auto &is_gpu : {false, true})
        {
            for (const auto &framework : frameworks)
            {
                if (is_gpu && !framework.has_gpu_version)
                {
                    // This framework doesn't have a GPU specific version
                    continue;
                }

                for (const auto &version : framework.versions)
                {
                    BackendLoadSpec item;
                    item.type    = framework.type;
                    item.version = version;

                    if (is_absolute_path)
                    {
                        // Ex:
                        //  "/usr/local/lib/neuropod/0.2.0/backends/torchscript_1.4.0/libneuropod_torchscript_backend.so"
                        // Ex:
                        //  "/usr/local/lib/neuropod/0.2.0/backends/torchscript_1.4.0_gpu/libneuropod_torchscript_backend.so"
                        item.path = neuropod_base_dir + ("/" NEUROPOD_VERSION "/backends/") + framework.type + "_" +
                                    version + (is_gpu ? "_gpu" : "") + "/" + framework.soname;
                    }
                    else
                    {
                        item.path = framework.soname;
                    }

                    out.emplace_back(std::move(item));
                }
            }
        }
    }

    return out;
}

// A list of backends, versions, and their corresponding so files.
// These will only be loaded if a backend for the requested type hasn't already been loaded.
// Note: these are listed in reverse priority order
const std::vector<BackendLoadSpec> default_backend_for_type = get_default_backend_map();

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

            continue;
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

// Loading process:
// 1. If we already have a registered backend that matches the type and target version range, return it
// 2. If we already have a registered backend that matches the type, throw an error explaining that OPE
//    is required
// 3. Attempt to load a backend given the user provided list of overrides
// 4. If that failed, attempt to load a backend with the default backend for type map above
// 5. Now that we (likely) loaded a backend, try again to find a registered backend that matches the request and return
// it
// 6. Throw an error with installation instructions
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
    NEUROPOD_ERROR(
        "The model being loaded requires a Neuropod backend for type '{}' and version range '{}'. However, "
        "a backend satisfying these requirements was not found. See the installation instructions "
        "at https://neuropod.ai/installing to install a backend. Retry with log level TRACE for more information.",
        type,
        target_version_range);
}

} // namespace neuropod
