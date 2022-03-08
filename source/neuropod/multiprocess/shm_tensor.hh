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

#pragma once

#include "neuropod/backends/tensor_allocator.hh"
#include "neuropod/internal/deleter.hh"
#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/neuropod_tensor.hh"
#include "neuropod/multiprocess/serialization/ipc_serialization.hh"
#include "neuropod/multiprocess/shm/shm_allocator.hh"

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
            NEUROPOD_ERROR("For the multiprocess backend, tensors must have less than {} dimensions. Tried creating "
                           "tensor with {} dimensions",
                           MAX_DIMS,
                           data_->ndims);
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

    void overwrite_type(TensorType type) { data_->tensor_type = type; }

    // Get a pointer to the underlying data
    void *get_untyped_data_ptr() { return data_->data; }

    const void *get_untyped_data_ptr() const { return data_->data; }

    SHMBlockID get_native_data() { return block_id_; };
};

std::shared_ptr<NeuropodTensor> tensor_from_id(const SHMBlockID &block_id);

inline std::vector<int64_t> copy_and_strip_last_dim(std::vector<int64_t> vec)
{
    vec.pop_back();
    return vec;
}

// Specialization for strings
namespace
{

// A struct that is used to represent elements in string tensors
// See below for more detail
struct __attribute__((__packed__)) string_wrapper
{
    // The length of the string (in bytes)
    int64_t length;

    // The string data
    uint8_t data[];
};

} // namespace

// This tensor operates on a local vector and then copies everything into shared memory when `get_native_data` is called
// For now, we only call `get_native_data` once per tensor so this should be okay
// TODO(vip): Optimize
// It's hard to make this zero-copy because of different string representations and variable string lengths
// Even so, this implementation makes more copies than necessary
// A more optimal implementation would use dynamically growing backing buffers in shared memory
// with a header pointing to offsets within the buffer for different elements
template <>
class SHMNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>, public NativeDataContainer<SHMBlockID>
{
private:
    std::vector<std::string> write_buffer_;

    // This is the last shm block we created (if any)
    std::unique_ptr<SHMNeuropodTensor<uint8_t>> last_shm_block_;

public:
    SHMNeuropodTensor(const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<std::string>(dims), write_buffer_(this->get_num_elements())
    {
    }

    // Load an existing tensor
    // See tensor_from_id
    SHMNeuropodTensor(const std::vector<int64_t> &dims,
                      std::shared_ptr<void>       block,
                      shm_tensor *                data,
                      const SHMBlockID &          block_id)
        : TypedNeuropodTensor<std::string>(copy_and_strip_last_dim(dims)), write_buffer_(this->get_num_elements())
    {
        auto byte_block = stdx::make_unique<SHMNeuropodTensor<uint8_t>>(dims, std::move(block), data, block_id);

        auto base_ptr = byte_block->get_raw_data_ptr();
        auto max_len  = dims[dims.size() - 1];

        // Copy into our local buffer
        auto numel = get_num_elements();
        for (int i = 0; i < numel; i++)
        {
            auto pos     = base_ptr + i * max_len;
            auto wrapper = reinterpret_cast<string_wrapper *>(pos);

            write_buffer_[i] = std::string(wrapper->data, wrapper->data + wrapper->length);
        }
    }

    ~SHMNeuropodTensor() = default;

    void copy_from(const std::vector<std::string> &vec) { write_buffer_ = vec; }

    SHMBlockID get_native_data()
    {
        // Compute the last dim size
        size_t max_len = 0;
        for (const auto &item : write_buffer_)
        {
            max_len = std::max(max_len, item.size());
        }

        // Space for the metadata in the wrapper struct
        max_len += sizeof(string_wrapper);

        // Add it to dims
        auto dims_copy = get_dims();
        dims_copy.push_back(max_len);

        // TODO(vip): We can optimize this
        last_shm_block_ = stdx::make_unique<SHMNeuropodTensor<uint8_t>>(dims_copy);
        last_shm_block_->overwrite_type(STRING_TENSOR);

        // Copy data in
        auto pos = last_shm_block_->get_raw_data_ptr();
        for (const auto &item : write_buffer_)
        {
            auto wrapper = reinterpret_cast<string_wrapper *>(pos);

            // Set the item size
            wrapper->length = item.size();

            // Copy in the data
            std::copy(item.begin(), item.end(), wrapper->data);

            // Move the pointer
            pos += max_len;
        }

        return last_shm_block_->get_native_data();
    };

protected:
    std::string get(size_t index) const { return write_buffer_.at(index); }

    void set(size_t index, const std::string &value) { write_buffer_[index] = value; }
};

// Serialization specializations for SHMNeuropodTensor
// Note: the specialization is for `shared_ptr<NeuropodValue>`, but we check internally
// that the item is a SHMNeuropodTensor
template <>
void ipc_serialize(std::ostream &out, const std::shared_ptr<NeuropodValue> &item);

template <>
void ipc_deserialize(std::istream &in, std::shared_ptr<NeuropodValue> &item);

} // namespace neuropod
