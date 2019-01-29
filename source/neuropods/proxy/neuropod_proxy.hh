//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/fwd_declarations.hh"

namespace neuropods
{

// A NeuropodProxy is just a NeuropodBackend that happens to be proxy
// This alias is just so the `Neuropod` interface is less confusing to users
using NeuropodProxy = NeuropodBackend;
}; // namespace neuropods
