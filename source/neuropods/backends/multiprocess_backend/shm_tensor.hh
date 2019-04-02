//
// Uber, Inc. (c) 2019
//

#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/internal/serialized_tensor.hh"

namespace neuropods
{

namespace ipc = boost::interprocess;

namespace
{
struct shm_wrapper
{
    // A mutex to make sure that we correctly keep track of the refcount
    ipc::interprocess_mutex mutex;

    // The number of unique references to this chunk of memory
    // (across processes)
    size_t process_refcount = 0;

    // The serialized tensor
    serialized_wrapper tensor;
};

// Increments the refcount for a chunk of memory
void increment_refcount(shm_wrapper * header)
{
    ipc::scoped_lock<ipc::interprocess_mutex> lock(header->mutex);
    header->process_refcount++;
}

// Decrements the refcount for a chunk of memory and deletes it
// if necessary
void decrement_refcount(shm_wrapper * header, const std::string &uuid)
{
    ipc::scoped_lock<ipc::interprocess_mutex> lock(header->mutex);
    header->process_refcount--;

    if (header->process_refcount == 0)
    {
        // Delete the shared memory region
        ipc::shared_memory_object::remove(("neuropod." + uuid).c_str());
    }
}

// Load a serialized_wrapper from a chunk of shared memory
// and correctly handle refcounting
std::shared_ptr<serialized_wrapper> get_serialized_wrapper_from_shm(
    std::shared_ptr<ipc::shared_memory_object> shm,
    const std::string &uuid)
{
    // mmap the shared memory
    auto region = std::make_shared<ipc::mapped_region>(*shm, ipc::read_write);
    auto wrapper = static_cast<shm_wrapper *>(region->get_address());

    // Increment the refcount
    increment_refcount(wrapper);
    return std::shared_ptr<serialized_wrapper>(&wrapper->tensor, [shm, region, wrapper, uuid](serialized_wrapper *p) {
        // Unmap and close when we're done with this data (destructors of shm and region)
        // Also free the shared memory if no other processes are using it
        decrement_refcount(wrapper, uuid);
    });
}

} // namespace

// A NeuropodTensor where the data is stored in shared memory
// This is a wrapper around SerializedNeuropodTensor with a custom allocator
template <typename T>
class SHMNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<std::string>
{
public:
    SHMNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<T>(name, dims),
          uuid_(boost::uuids::to_string(boost::uuids::random_generator()()))
    {
        std::string uuid = uuid_;

        // An allocator that allocates in shared memory
        SerializedWrapperAllocator allocator = [uuid](size_t length) {
            // Create a shared memory object
            auto shm = std::make_shared<ipc::shared_memory_object>(ipc::create_only, ("neuropod." + uuid).c_str(), ipc::read_write);

            // Set the size of the shared memory
            // TODO(vip): this allocates more memory than required
            shm->truncate(length + sizeof(shm_wrapper));

            return get_serialized_wrapper_from_shm(shm, uuid);
        };

        serialized_tensor_ = stdx::make_unique<SerializedNeuropodTensor<T>>(name, dims, allocator);
    }

    // This tensor type doesn't support wrapping existing memory so we'll do a copy
    SHMNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims, void * data, const Deleter &deleter)
        : SHMNeuropodTensor<T>(name, dims)
    {
        // We need to make a copy in order to serialize
        this->copy_from(static_cast<T *>(data), this->get_num_elements());

        // Make sure we run the deleter
        run_deleter(register_deleter(deleter, data));
    }

    // Load an already created shm tensor
    SHMNeuropodTensor(
        const std::string &name,
        const std::vector<int64_t> &dims,
        std::shared_ptr<serialized_wrapper> serialized,
        const std::string &uuid
    )
        : TypedNeuropodTensor<T>(name, dims), uuid_(uuid)
    {
        serialized_tensor_ = stdx::make_unique<SerializedNeuropodTensor<T>>(name, dims, serialized);
    }

    ~SHMNeuropodTensor() = default;

    // Get a pointer to the underlying data
    T *get_raw_data_ptr() {
        return serialized_tensor_->get_raw_data_ptr();
    }

    const T *get_raw_data_ptr() const {
        return serialized_tensor_->get_raw_data_ptr();
    }

    std::string get_native_data() {
        // Return an shm_key used to load the tensor in the other process
        return uuid_;
    }

private:
    // Handles the serialization for us
    std::unique_ptr<SerializedNeuropodTensor<T>> serialized_tensor_;

    // A uuid used to identify the shared memory blob
    std::string uuid_;
};

// A function that loads a tensor given a shm_key
std::shared_ptr<NeuropodTensor> tensor_from_shm_key(const std::string &shm_key)
{
    // Get the length and the uuid from the shm key
    auto uuid = shm_key.substr(shm_key.find(",") + 1);

    // Create a shared memory object
    auto shm = std::make_shared<ipc::shared_memory_object>(ipc::open_only, ("neuropod." + uuid).c_str(), ipc::read_write);

    auto wrapper = get_serialized_wrapper_from_shm(shm, uuid);

    // Make a SHMNeuropodTensor
    return deserialize_tensor<SHMNeuropodTensor>(wrapper, uuid);
}

} // namespace neuropods
