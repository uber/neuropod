//
// Uber, Inc. (c) 2018
//

#include "backend_registration.hh"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace neuropods
{

namespace
{

std::unordered_map<std::string, BackendFactoryFunction> registered_backends_by_type;
std::unordered_map<std::string, BackendFactoryFunction> registered_backends_by_name;

} // namespace

bool register_backend(const std::string &             name,
                      const std::vector<std::string> &supported_types,
                      BackendFactoryFunction          factory_fn)
{
    for (const std::string &type : supported_types)
    {
        registered_backends_by_type[type] = factory_fn;
    }

    registered_backends_by_name[name] = factory_fn;

    return true;
}

BackendFactoryFunction get_backend_for_type(const std::string &type)
{
    auto backend_it = registered_backends_by_type.find(type);
    if (backend_it == registered_backends_by_type.end())
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
    auto backend_it = registered_backends_by_name.find(name);
    if (backend_it == registered_backends_by_name.end())
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
