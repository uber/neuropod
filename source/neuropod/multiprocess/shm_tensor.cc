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

// Serialization specializations for SHMNeuropodTensor and SealedSHMTensor
template <>
void ipc_serialize(std::ostream &out, const NativeDataContainer<SHMBlockID> &data)
{
    ipc_serialize(out, data.get_native_data());
}

template <>
void ipc_serialize(std::ostream &out, const SealedSHMTensor &data)
{
    ipc_serialize(out, data.block_id);
}

// Serialization specializations for SHMNeuropodTensor
// Note: the specialization is for `shared_ptr<NeuropodValue>`, but we check internally
// that the item is a SHMNeuropodTensor or a SealedSHMTensor
template <>
void ipc_serialize(std::ostream &out, const std::shared_ptr<NeuropodValue> &data)
{
    // TODO(vip): Do this in a better way
    if (auto container = dynamic_cast<SealedSHMTensor *>(data.get()))
    {
        // SealedSHMTensor
        ipc_serialize(out, *container);
    }
    else if (auto container = dynamic_cast<NativeDataContainer<SHMBlockID> *>(data.get()))
    {
        // SHMNeuropodTensor
        ipc_serialize(out, *container);
    }
    else
    {
        NEUROPOD_ERROR(
            "ipc_serialize only works with NeuropodValueMaps containing SHMNeuropodTensor or SealedSHMTensors. The "
            "supplied map contained tensors of another type.");
    }
}

// Only one deserialize implementation
template <>
void ipc_deserialize(std::istream &in, std::shared_ptr<NeuropodValue> &data)
{
    // Read the block ID
    SHMBlockID block_id;
    detail::checked_read(in, reinterpret_cast<char *>(block_id.data()), block_id.size());

    // Load the tensor
    data = tensor_from_id(block_id);
}

} // namespace neuropod
