//
// Uber, Inc. (c) 2018
//

#include "backend_registration.hh"

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

    auto backend_it = registered_backends_by_type->find(type);
    if (backend_it == registered_backends_by_type->end())
    {
        std::stringstream ss;
        ss << "Neuropod backend not found for type '" << type << "'! ";
        ss << "Make sure that you have a build dependency on the correct backend";
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
