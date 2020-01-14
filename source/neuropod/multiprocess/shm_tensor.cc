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

} // namespace neuropod
