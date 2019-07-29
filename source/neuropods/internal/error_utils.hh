//
// Uber, Inc. (c) 2019
//

#pragma once

#include <iostream>
#include <sstream>

namespace neuropods
{

// A helper macro that lets us do things like
// NEUROPOD_ERROR("Expected value " << a << ", but got " << b)
#define NEUROPOD_ERROR(MSG)                  \
    {                                        \
        std::stringstream err;               \
        err << "Neuropod Error: ";           \
        err << MSG;                          \
        throw std::runtime_error(err.str()); \
    }

} // namespace neuropods
