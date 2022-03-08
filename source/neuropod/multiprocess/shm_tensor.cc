/* Copyright (c) 2020 The Neuropod Authors

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

#include "neuropod/multiprocess/shm_tensor.hh"

namespace neuropod
{

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

// Serialization specializations for SHMNeuropodTensor
// Note: the specialization is for `shared_ptr<NeuropodValue>`, but we check internally
// that the item is a SHMNeuropodTensor
template <>
void ipc_serialize(std::ostream &out, const std::shared_ptr<NeuropodValue> &item)
{
    // Cast to a `NativeDataContainer`
    auto container = std::dynamic_pointer_cast<NativeDataContainer<SHMBlockID>>(item);
    if (!container)
    {
        NEUROPOD_ERROR("ipc_serialize only works with NeuropodValueMaps containing SHMNeuropodTensors. The "
                       "supplied map contained tensors of another type.");
    }

    // Write the block ID
    const auto &block_id = container->get_native_data();

    detail::checked_write(out, reinterpret_cast<const char *>(block_id.data()), block_id.size());
}

template <>
void ipc_deserialize(std::istream &in, std::shared_ptr<NeuropodValue> &item)
{
    // Read the block ID
    SHMBlockID block_id;
    detail::checked_read(in, reinterpret_cast<char *>(block_id.data()), block_id.size());

    // Load the tensor
    item = tensor_from_id(block_id);
}

} // namespace neuropod
