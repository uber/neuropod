//
// Uber, Inc. (c) 2020
//

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
