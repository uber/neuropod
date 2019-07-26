//
// Uber, Inc. (c) 2019
//

#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <memory>
#include <mutex>
#include <vector>

#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/multiprocess/shm_allocator.hh"

namespace neuropods
{

namespace
{

constexpr size_t MAX_DIMS = 128;

struct __attribute__((__packed__)) shm_tensor
{
    TensorType tensor_type;

    uint64_t ndims;
    int64_t  dims[MAX_DIMS];

    uint8_t  data[];
};

// TODO(vip): Use std::align
void * get_next_aligned_offset(void * base)
{
    // We want to find an offset such that the data will be 64 byte aligned
    uint64_t base_address = reinterpret_cast<uint64_t>(base);
    size_t aligned_offset = 64 - (base_address + sizeof(shm_tensor)) % 64;
    return reinterpret_cast<void *>(base_address + aligned_offset);
}

} // namespace

template <typename T>
class SHMNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<boost::uuids::uuid&>
{
private:
    // A pointer to the block of shared memory
    std::shared_ptr<void> block_;

    // A pointer to the data contained in the tensor
    shm_tensor *data_;

    // The tensor's UUID
    boost::uuids::uuid uuid_;

public:
    SHMNeuropodTensor(const std::vector<int64_t> &dims) : TypedNeuropodTensor<T>(dims)
    {
        // Give us room to make sure that everything is 64 byte aligned
        size_t size_bytes = sizeof(shm_tensor) + this->get_num_elements() * sizeof(T) + 64;

        // Get a block of shared memory
        block_ = get_shm(size_bytes, uuid_);

        // Get a pointer to the struct and initialize it
        // (make sure it's 64 byte aligned)
        data_ = new (get_next_aligned_offset(block_.get())) shm_tensor;

        if (reinterpret_cast<uint64_t>(data_->data) % 64 != 0)
        {
            NEUROPOD_ERROR("Expected data to be aligned!")
        }

        // Set all the metadata
        data_->tensor_type = this->get_tensor_type();
        data_->ndims = dims.size();
        if (data_->ndims >= MAX_DIMS)
        {
            NEUROPOD_ERROR("For the multiprocess backend, tensors must have less than " << MAX_DIMS << " dimensions. Tried creating tensor with " << data_->ndims << " dimensions")
        }

        std::copy(dims.begin(), dims.end(), data_->dims);
    }

    // Load an existing tensor
    // See tensor_from_shm_key
    SHMNeuropodTensor(const std::vector<int64_t> &dims,
                      std::shared_ptr<void> block,
                      shm_tensor *data,
                      const boost::uuids::uuid &uuid)
        : TypedNeuropodTensor<T>(dims), block_(block), data_(data), uuid_(uuid)
    {
        if (reinterpret_cast<uint64_t>(data_->data) % 64 != 0)
        {
            NEUROPOD_ERROR("Expected data to be aligned!")
        }
    }


    // This backend cannot wrap existing memory so we need to make a copy
    SHMNeuropodTensor(const std::vector<int64_t> &dims, void * data, const Deleter &deleter)
        : SHMNeuropodTensor<T>(dims)
    {
        // Copy in the data
        this->copy_from(static_cast<T *>(data), this->get_num_elements());

        // Run the deleter
        run_deleter(register_deleter(deleter, data));
    }

    ~SHMNeuropodTensor() = default;

    // Get a pointer to the underlying data
    T *get_raw_data_ptr() { return reinterpret_cast<T *>(data_->data); }

    const T *get_raw_data_ptr() const { return reinterpret_cast<T *>(data_->data); }

    boost::uuids::uuid& get_native_data() { return uuid_; };
};

std::shared_ptr<NeuropodTensor> tensor_from_uuid(const boost::uuids::uuid &uuid)
{
    // Load the block of shared memory
    auto block = load_shm(uuid);

    // Get a pointer to the struct
    auto data = static_cast<shm_tensor *>(get_next_aligned_offset(block.get()));

    // Get the number of dims
    std::vector<int64_t> dims(data->dims, data->dims + data->ndims);

    return make_tensor_no_string<SHMNeuropodTensor>(data->tensor_type, dims, std::move(block), std::move(data), uuid);
}

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

    void set(const std::vector<std::string> &data)
    {
        data_ = data;
    }

    std::vector<std::string> get_data_as_vector() const
    {
        return data_;
    }
};

} // namespace neuropods
