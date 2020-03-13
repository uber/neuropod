//
// Uber, Inc. (c) 2020
//

#include "neuropod/multiprocess/serialization/ipc_serialization.hh"

namespace neuropod
{

// Specialization for uint64_t (for DONE messages)
template <>
void ipc_serialize(std::ostream &out, const uint64_t &data)
{
    out.write(reinterpret_cast<const char *>(&data), sizeof(data));
}

template <>
void ipc_deserialize(std::istream &in, uint64_t &data)
{
    in.read(reinterpret_cast<char *>(&data), sizeof(data));
}

} // namespace neuropod
