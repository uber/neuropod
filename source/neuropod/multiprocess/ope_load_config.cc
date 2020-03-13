//
// Uber, Inc. (c) 2020
//

#include "neuropod/multiprocess/ope_load_config.hh"

namespace neuropod
{

// Specializations for ope_load_config
// Because of a bug in boost serialization, we can't just mark that struct
// as serializable
template <>
void ipc_serialize(std::ostream &out, const ope_load_config &data)
{
    boost::archive::binary_oarchive ar{out};
    ar << data.neuropod_path;
}

template <>
void ipc_deserialize(std::istream &in, ope_load_config &data)
{
    boost::archive::binary_iarchive ar{in};
    ar >> data.neuropod_path;
}

} // namespace neuropod
