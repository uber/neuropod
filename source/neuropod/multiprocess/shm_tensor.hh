//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropod/backends/tensor_allocator.hh"
#include "neuropod/internal/deleter.hh"
#include "neuropod/internal/neuropod_tensor.hh"
#include "neuropod/multiprocess/shm_allocator.hh"

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace neuropod
{

namespace
{

constexpr size_t MAX_DIMS = 128;

struct __attribute__((__packed__)) shm_tensor
{
    TensorType tensor_type;

    uint64_t ndims;
    int64_t  dims[MAX_DIMS];

    uint8_t data[];
};

// TODO(vip): Use std::align
inline void *get_next_aligned_offset(void *base)
{
    // We want to find an offset such that the data will be 64 byte aligned
    uint64_t base_address   = reinterpret_cast<uint64_t>(base);
    size_t   aligned_offset = 64 - (base_address + sizeof(shm_tensor)) % 64;
    return reinterpret_cast<void *>(base_address + aligned_offset);
}

} // namespace

// A shared memory allocator
extern SHMAllocator shm_allocator;

template <typename T>
class SHMNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<SHMBlockID>
{
private:
    // A pointer to the block of shared memory
    std::shared_ptr<void> block_;

    // A pointer to the data contained in the tensor
    shm_tensor *data_;

    // The ID of the chunk of shared memory
    SHMBlockID block_id_;

public:
    SHMNeuropodTensor(const std::vector<int64_t> &dims) : TypedNeuropodTensor<T>(dims)
    {
        // Give us room to make sure that everything is 64 byte aligned
        const size_t size_bytes = sizeof(shm_tensor) + this->get_num_elements() * sizeof(T) + 64;

        // Get a block of shared memory
        block_ = shm_allocator.allocate_shm(size_bytes, block_id_);

        // Get a pointer to the struct and initialize it
        data_ = new (get_next_aligned_offset(block_.get())) shm_tensor;

        // Make sure it's 64 byte aligned
        assert(reinterpret_cast<uint64_t>(data_->data) % 64 == 0);

        // Set all the metadata
        data_->tensor_type = this->get_tensor_type();
        data_->ndims       = dims.size();
        if (data_->ndims >= MAX_DIMS)
        {
            NEUROPOD_ERROR("For the multiprocess backend, tensors must have less than "
                           << MAX_DIMS << " dimensions. Tried creating tensor with " << data_->ndims << " dimensions")
        }

        std::copy(dims.begin(), dims.end(), data_->dims);
    }

    // Load an existing tensor
    // See tensor_from_id
    SHMNeuropodTensor(const std::vector<int64_t> &dims,
                      std::shared_ptr<void>       block,
                      shm_tensor *                data,
                      const SHMBlockID &          block_id)
        : TypedNeuropodTensor<T>(dims), block_(block), data_(data), block_id_(block_id)
    {
        // Make sure data is 64 byte aligned
        assert(reinterpret_cast<uint64_t>(data_->data) % 64 == 0);
    }

    // This backend cannot wrap existing memory so we need to make a copy
    SHMNeuropodTensor(const std::vector<int64_t> &dims, void *data, const Deleter &deleter) : SHMNeuropodTensor<T>(dims)
    {
        // Copy in the data
        this->copy_from(static_cast<T *>(data), this->get_num_elements());

        // Since we made a copy of the data, we no longer need to keep the original
        // "alive". Run the deleter to let the user know that they can dispose of the
        // data if they want
        run_deleter(register_deleter(deleter, data));
    }

    ~SHMNeuropodTensor() = default;

    // Get a pointer to the underlying data
    void *get_untyped_data_ptr() { return data_->data; }

    const void *get_untyped_data_ptr() const { return data_->data; }

    SHMBlockID get_native_data() { return block_id_; };
};

std::shared_ptr<NeuropodTensor> tensor_from_id(const SHMBlockID &block_id);

// Specialization for strings
// TODO(vip): implement
template <>
class SHMNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>
{
private:
    // The data contained in the tensor
    std::vector<std::string> data_;

public:
    SHMNeuropodTensor(const std::vector<int64_t> &dims) : TypedNeuropodTensor<std::string>(dims)
    {
        NEUROPOD_ERROR("String tensors are not yet supported for multiprocess execution")
    }

    ~SHMNeuropodTensor() = default;

    void set(const std::vector<std::string> &data) { data_ = data; }

protected:
    const std::string operator[](size_t index) const { return data_[index]; }
};

} // namespace neuropod
