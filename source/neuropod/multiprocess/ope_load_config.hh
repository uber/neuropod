/* Copyright (c) 2020 UATC, LLC

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

#include "neuropod/internal/backend_registration.hh"
#include "neuropod/multiprocess/serialization/ipc_serialization.hh"

namespace neuropod
{

// Contains everything needed to load a model in the worker process
struct ope_load_config
{
    // The path of the model to load
    std::string neuropod_path;

    // See the docs in `neuropod.hh`
    std::vector<BackendLoadSpec> default_backend_overrides;
};

// Serialization specializations for ope_load_config
template <>
void ipc_serialize(std::ostream &out, const ope_load_config &data);

template <>
void ipc_deserialize(std::istream &in, ope_load_config &data);

} // namespace neuropod
