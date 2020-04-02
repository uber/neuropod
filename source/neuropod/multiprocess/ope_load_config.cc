//
// Uber, Inc. (c) 2020
//

#include "neuropod/multiprocess/ope_load_config.hh"

namespace neuropod
{

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
