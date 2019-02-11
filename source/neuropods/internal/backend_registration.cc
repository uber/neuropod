//
// Uber, Inc. (c) 2018
//

#include "backend_registration.hh"

#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace neuropods
{

namespace
{

std::once_flag registrar_initialized;
std::unordered_map<std::string, BackendFactoryFunction>* registered_backends_by_type = nullptr;
std::unordered_map<std::string, BackendFactoryFunction>* registered_backends_by_name = nullptr;

void init_registrar_if_needed()
{
    std::call_once(registrar_initialized, [](){
        registered_backends_by_name = new std::unordered_map<std::string, BackendFactoryFunction>();
        registered_backends_by_type = new std::unordered_map<std::string, BackendFactoryFunction>();
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

void load_default_backend(const std::string &type)
{
    auto backend_it = default_backend_for_type.find(type);
    if (backend_it == default_backend_for_type.end())
    {
        std::stringstream ss;
        ss << "Default Neuropod backend not found for type '" << type << "'! ";
        ss << "Make sure that you load a Neuropod backend that can support '" << type << "'";
        throw std::runtime_error(ss.str());
    }
    else
    {
        if (dlopen(backend_it->second.c_str(), RTLD_NOW) == nullptr)
        {
            std::stringstream ss;
            ss << "Loading the default backend for type '" << type << "' failed. ";
            ss << "Error from dlopen: " << dlerror();
            throw std::runtime_error(ss.str());
        }
    }
}

} // namespace

bool register_backend(const std::string &             name,
                      const std::vector<std::string> &supported_types,
                      BackendFactoryFunction          factory_fn)
{
    init_registrar_if_needed();

    for (const std::string &type : supported_types)
    {
        (*registered_backends_by_type)[type] = factory_fn;
    }

    (*registered_backends_by_name)[name] = factory_fn;

    return true;
}

BackendFactoryFunction get_backend_for_type(const std::string &type)
{
    init_registrar_if_needed();

    if (registered_backends_by_type->find(type) == registered_backends_by_type->end())
    {
        // We don't have a backend loaded for the requested type
        // Try loading the default one
        load_default_backend(type);
    }

    auto backend_it = registered_backends_by_type->find(type);
    if (backend_it == registered_backends_by_type->end())
    {
        // If we get here, that means that we tried loading a default backend
        // and it failed
        std::stringstream ss;
        ss << "Neuropod backend not found for type '" << type << "'! ";
        ss << "Loading the default backend for type '" << type << "' failed.";
        throw std::runtime_error(ss.str());
    }
    else
    {
        return backend_it->second;
    }
}

BackendFactoryFunction get_backend_by_name(const std::string &name)
{
    init_registrar_if_needed();

    auto backend_it = registered_backends_by_name->find(name);
    if (backend_it == registered_backends_by_name->end())
    {
        std::stringstream ss;
        ss << "Neuropod backend not found for name'" << name << "'! ";
        ss << "Make sure that you have a build dependency on the correct backend";
        throw std::runtime_error(ss.str());
    }
    else
    {
        return backend_it->second;
    }
}

} // namespace neuropods
