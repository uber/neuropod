//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropod/backends/tensor_allocator.hh"
#include "neuropod/internal/deleter.hh"
#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/neuropod_tensor.hh"
#include "neuropod/multiprocess/ipc_control_channel.hh"
#include "neuropod/multiprocess/serialization/ipc_serialization.hh"
#include "neuropod/multiprocess/shm/shm_allocator.hh"

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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

// A sealed SHM tensor
class SealedSHMTensor : public SealedNeuropodTensor
{
public:
    // A pointer to the block of shared memory
    std::shared_ptr<void> block;

    // The ID of the chunk of shared memory
    SHMBlockID block_id;

    // The device to seal with in the worker process
    NeuropodDevice device;
};

// Batches tensors and seals them together
class SHMSealer
{
private:
    std::vector<std::shared_ptr<SealedSHMTensor>> items;
    std::shared_ptr<IPCControlChannel>            control_channel_;
    std::condition_variable                       cv_;
    std::mutex                                    mutex_;

    std::mutex  send_mutex_;
    bool        shutdown_ = false;
    std::thread thread_;

    void worker_loop()
    {
        while (!shutdown_)
        {
            std::vector<std::shared_ptr<SealedSHMTensor>> temp;
            std::unique_lock<std::mutex>                  lock(mutex_);
            if (items.empty())
            {
                cv_.wait(lock, [&] { return shutdown_ || !items.empty(); });
            }

            if (shutdown_)
            {
                break;
            }

            // Swap so we can unlock the mutex as soon as possible
            std::lock_guard<std::mutex> send_lock(send_mutex_);
            std::swap(temp, items);
            lock.unlock();

            // Send the message
            control_channel_->send_message_move(SEAL, std::move(temp));
        }
    }

public:
    SHMSealer(std::shared_ptr<IPCControlChannel> control_channel)
        : control_channel_(control_channel), thread_(&SHMSealer::worker_loop, this)
    {
    }
    ~SHMSealer()
    {
        {
            // Lock the mutex
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

    void sync_flush_buffers()
    {
        // Lock the mutex
        std::lock_guard<std::mutex> lock(mutex_);

        // We lock this mutex too to make sure nothing is in flight
        // When `sync_flush_buffers` returns, all items that were in the buffer
        // before this method was called should have been sent already.
        // If we don't lock this mutex as well, it's possible that the worker thread
        // is in the process of sending items that it removed from the buffer
        std::lock_guard<std::mutex> send_lock(send_mutex_);

        if (items.empty())
        {
            return;
        }

        // Send the message
        control_channel_->send_message_move(SEAL, std::move(items));

        // Return `items` to a valid state after the `std::move`
        items.clear();
    }

    void seal(std::shared_ptr<SealedSHMTensor> item)
    {
        bool notify = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            items.emplace_back(std::move(item));
            if (items.size() == 1)
            {
                notify = true;
            }
        }

        if (notify)
        {
            cv_.notify_all();
        }
    }
};

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

    // The control channel to send seal messages on
    std::shared_ptr<SHMSealer> shm_sealer_;

public:
    SHMNeuropodTensor(const std::vector<int64_t> &dims, std::shared_ptr<SHMSealer> shm_sealer = nullptr)
        : TypedNeuropodTensor<T>(dims), shm_sealer_(std::move(shm_sealer))
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
    SHMNeuropodTensor(const std::vector<int64_t> &dims,
                      void *                      data,
                      const Deleter &             deleter,
                      std::shared_ptr<SHMSealer>  shm_sealer = nullptr)
        : SHMNeuropodTensor<T>(dims)
    {
        // Set the control channel
        shm_sealer_ = std::move(shm_sealer);

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

    SHMBlockID get_native_data() const { return block_id_; };

    std::shared_ptr<SealedNeuropodTensor> seal(NeuropodDevice device)
    {
        // Create the sealed SHM tensor
        auto out = std::make_shared<SealedSHMTensor>();

        out->block    = block_;
        out->block_id = block_id_;
        out->device   = device;

        if (shm_sealer_)
        {
            shm_sealer_->seal(out);
        }

        return out;
    }
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

// TODO(vip): Optimize
template <>
class SHMNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>, public NativeDataContainer<SHMBlockID>
{
private:
    // We store N dimensional string tensors as N + 1 dimensional char tensors
    // where the last dimension is the size of the max length string + 8 bytes
    // The first 8 bytes are used to store the size of the string.
    std::unique_ptr<SHMNeuropodTensor<uint8_t>> data_;

    // The length of the longest string in this tensor
    size_t max_len_;

    // The control channel to send seal messages on
    std::shared_ptr<SHMSealer> sealer_;

public:
    SHMNeuropodTensor(const std::vector<int64_t> &dims, std::shared_ptr<SHMSealer> sealer = nullptr)
        : TypedNeuropodTensor<std::string>(dims), max_len_(0), sealer_(std::move(sealer))
    {
    }

    // Load an existing tensor
    // See tensor_from_id
    SHMNeuropodTensor(const std::vector<int64_t> &dims,
                      std::shared_ptr<void>       block,
                      shm_tensor *                data,
                      const SHMBlockID &          block_id)
        : TypedNeuropodTensor<std::string>(copy_and_strip_last_dim(dims))
    {
        data_    = stdx::make_unique<SHMNeuropodTensor<uint8_t>>(dims, std::move(block), data, block_id);
        max_len_ = dims[dims.size() - 1];
    }

    ~SHMNeuropodTensor() = default;

    void set(const std::vector<std::string> &data)
    {
        // Compute the last dim size
        max_len_ = 0;
        for (const auto &item : data)
        {
            max_len_ = std::max(max_len_, item.size());
        }

        // Space for the metadata in the wrapper struct
        max_len_ += sizeof(string_wrapper);

        // Add it to dims
        auto dims_copy = get_dims();
        dims_copy.push_back(max_len_);

        // TODO(vip): We can optimize this
        data_ = stdx::make_unique<SHMNeuropodTensor<uint8_t>>(dims_copy, sealer_);
        data_->overwrite_type(STRING_TENSOR);

        // Copy data in
        auto pos = data_->get_raw_data_ptr();
        for (const auto &item : data)
        {
            auto wrapper = reinterpret_cast<string_wrapper *>(pos);

            // Set the item size
            wrapper->length = item.size();

            // Copy in the data
            std::copy(item.begin(), item.end(), wrapper->data);

            // Move the pointer
            pos += max_len_;
        }
    }

    SHMBlockID get_native_data() const { return data_->get_native_data(); };

protected:
    const std::string operator[](size_t index) const
    {
        auto pos     = data_->get_raw_data_ptr() + index * max_len_;
        auto wrapper = reinterpret_cast<string_wrapper *>(pos);

        return std::string(wrapper->data, wrapper->data + wrapper->length);
    }

    std::shared_ptr<SealedNeuropodTensor> seal(NeuropodDevice device) { return data_->seal(device); }
};

// Serialization specializations for SHMBlockID
template <>
inline void ipc_serialize(std::ostream &out, const SHMBlockID &block_id)
{
    detail::checked_write(out, reinterpret_cast<const char *>(block_id.data()), block_id.size());
}

template <>
inline void ipc_deserialize(std::istream &in, SHMBlockID &block_id)
{
    detail::checked_read(in, reinterpret_cast<char *>(block_id.data()), block_id.size());
}

// Serialization specializations for SealedSHMTensor
template <>
inline void ipc_serialize(std::ostream &out, const SealedSHMTensor &data)
{
    ipc_serialize(out, data.block_id);
    ipc_serialize(out, data.device);
}

template <>
inline void ipc_deserialize(std::istream &in, SealedSHMTensor &data)
{
    ipc_deserialize(in, data.block_id);
    ipc_deserialize(in, data.device);
}

// Serialization specializations for SHMNeuropodTensor
// Note: the specialization is for `shared_ptr<NeuropodValue>`, but we check internally
// that the item is a SHMNeuropodTensor or a SealedSHMTensor
template <>
void ipc_serialize(std::ostream &out, const std::shared_ptr<NeuropodValue> &data);

template <>
void ipc_deserialize(std::istream &in, std::shared_ptr<NeuropodValue> &data);

} // namespace neuropod
