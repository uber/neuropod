/* Copyright (c) 2020 The Neuropod Authors

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
