//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropod/neuropod.hh"

#include <string>

namespace neuropod
{

// Load a neuropod using out of process execution.
// `default_backend_overrides` allows users to override the default backend for a given type.
// This is a mapping from a neuropod type (e.g. tensorflow, python, torchscript, etc.) to the
// name of a shared library that supports that type.
// Note: Libraries in this map will only be loaded if a backend for the requested type hasn't
// already been loaded
// See the comments in `neuropod.hh` for more details
std::unique_ptr<NeuropodBackend> load_neuropod_ope(const std::string &                 neuropod_path,
                                                   const RuntimeOptions &              options,
                                                   const std::vector<BackendLoadSpec> &default_backend_overrides);

} // namespace neuropod
