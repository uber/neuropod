//
// Uber, Inc. (c) 2019
//

#include "neuropod/multiprocess/shm_tensor.hh"

namespace neuropod
{

// The SHMAllocator used by all SHMNeuropodTensors
SHMAllocator shm_allocator;

std::shared_ptr<NeuropodTensor> tensor_from_id(const SHMBlockID &block_id)
{
    // Load the block of shared memory
    auto block = shm_allocator.load_shm(block_id);

    // Get a pointer to the struct
    auto data = static_cast<shm_tensor *>(get_next_aligned_offset(block.get()));

    // Get the number of dims
    std::vector<int64_t> dims(data->dims, data->dims + data->ndims);

    return make_tensor<SHMNeuropodTensor>(data->tensor_type, dims, std::move(block), data, block_id);
}

// Serialization specializations for a NeuropodValueMap of SHMTensors
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

} // namespace neuropod
