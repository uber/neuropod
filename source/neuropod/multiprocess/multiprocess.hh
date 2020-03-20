//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropod/neuropod.hh"

#include <string>

namespace neuropod
{

// Load a neuropod using out of process execution. See the comments in `RuntimeOptions` for more details
std::unique_ptr<NeuropodBackend> load_neuropod_ope(const std::string &neuropod_path, const RuntimeOptions &options);

} // namespace neuropod
