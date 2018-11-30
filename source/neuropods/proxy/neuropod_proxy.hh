//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/backends/neuropod_backend.hh"

namespace neuropods
{

// A NeuropodProxy is just a NeuropodBackend that happens to be proxy
// This alias is just so the `Neuropod` interface is less confusing to users
using NeuropodProxy = NeuropodBackend;
}; // namespace neuropods
