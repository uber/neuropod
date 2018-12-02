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

std::unordered_map<std::string, BackendFactoryFunction> registered_backends;

} // namespace

bool register_backend(const std::vector<std::string> &supported_types, BackendFactoryFunction factory_fn)
{
    for (const std::string &type : supported_types)
    {
        registered_backends[type] = factory_fn;
    }

    return true;
}

BackendFactoryFunction get_backend_for_type(const std::string &type)
{
    auto backend_it = registered_backends.find(type);
    if (backend_it == registered_backends.end())
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

} // namespace neuropods