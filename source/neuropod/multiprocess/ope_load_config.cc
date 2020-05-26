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

#include "neuropod/multiprocess/ope_load_config.hh"

namespace neuropod
{

// Specialization for BackendLoadSpec
template <>
inline void ipc_serialize(std::ostream &out, const BackendLoadSpec &data)
{
    ipc_serialize(out, data.type);
    ipc_serialize(out, data.version);
    ipc_serialize(out, data.path);
}

template <>
inline void ipc_deserialize(std::istream &in, BackendLoadSpec &data)
{
    ipc_deserialize(in, data.type);
    ipc_deserialize(in, data.version);
    ipc_deserialize(in, data.path);
}

template <>
void ipc_serialize(std::ostream &out, const ope_load_config &data)
{
    ipc_serialize(out, data.neuropod_path);
    ipc_serialize(out, data.default_backend_overrides);
}

template <>
void ipc_deserialize(std::istream &in, ope_load_config &data)
{
    ipc_deserialize(in, data.neuropod_path);
    ipc_deserialize(in, data.default_backend_overrides);
}

} // namespace neuropod
