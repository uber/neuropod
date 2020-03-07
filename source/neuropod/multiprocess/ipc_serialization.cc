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
    // Write the number of items
    size_t num_items = data.size();
    out.write(reinterpret_cast<const char *>(&num_items), sizeof(num_items));

    std::unordered_map<std::string, SHMBlockID> id_map;
    for (const auto &entry : data)
    {
        // Write the length of the name
        size_t name_len = entry.first.size();
        out.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));

        // Write the name
        out.write(entry.first.c_str(), name_len);

        // Write the block ID
        const auto &block_id =
            std::dynamic_pointer_cast<NativeDataContainer<SHMBlockID>>(entry.second)->get_native_data();

        out.write(reinterpret_cast<const char *>(block_id.data()), block_id.size());
    }
}

template <>
void ipc_deserialize(std::istream &in, NeuropodValueMap &data)
{
    // Read the number of items
    size_t num_items;
    in.read(reinterpret_cast<char *>(&num_items), sizeof(num_items));

    // Used for reading in tensor names
    char tmp[2048];

    for (int i = 0; i < num_items; i++)
    {
        // Read the length of the name
        size_t name_len;
        in.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));

        // Max len check and then read the name
        if (name_len > sizeof(tmp))
        {
            NEUROPOD_ERROR("For OPE, tensor names must be less than or equal to {} characters in length. Tried to read "
                           "a name with length {}",
                           sizeof(tmp),
                           name_len);
        }

        in.read(tmp, name_len);
        std::string name(tmp, name_len);

        // Read the block ID
        SHMBlockID block_id;
        in.read(reinterpret_cast<char *>(block_id.data()), block_id.size());

        // Make sure we're not overwriting anything
        auto &item = data[name];
        if (!item)
        {
            // Create a tensor
            item = tensor_from_id(block_id);
        }
        else
        {
            NEUROPOD_ERROR("When receiving a message as part of OPE, attempted to overwrite a tensor with name {}",
                           name);
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

// Specialization for uint64_t (for ACK messages)
// Serialization and deserialization are running on the same machine so we don't
// need to worry about things like endianness
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
