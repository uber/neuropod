//
// Uber, Inc. (c) 2020
//

#include "neuropod/multiprocess/ipc_serialization.hh"

#include "neuropod/multiprocess/shm_tensor.hh"

namespace neuropod
{

// Specializations for NeuropodValueMap
template <>
void ipc_serialize(std::ostream &out, const NeuropodValueMap &data)
{
    std::unordered_map<std::string, SHMBlockID> id_map;
    for (const auto &entry : data)
    {
        const auto &block_id =
            std::dynamic_pointer_cast<NativeDataContainer<SHMBlockID>>(entry.second)->get_native_data();

        id_map[entry.first] = block_id;
    }

    boost::archive::binary_oarchive ar{out};
    ar << id_map;
}

template <>
void ipc_deserialize(std::istream &in, NeuropodValueMap &data)
{
    std::unordered_map<std::string, SHMBlockID> id_map;
    boost::archive::binary_iarchive             ar{in};
    ar >> id_map;

    for (const auto &entry : id_map)
    {
        // Make sure we're not overwriting anything
        auto &item = data[entry.first];
        if (!item)
        {
            // Create a tensor
            item = tensor_from_id(entry.second);
        }
        else
        {
            NEUROPOD_ERROR("When receiving a message as part of OPE, attempted to overwrite a tensor with name {}",
                           entry.first);
        }
    }
}

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
