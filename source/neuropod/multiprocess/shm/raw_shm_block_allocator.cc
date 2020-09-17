/* Copyright (c) 2020 UATC, LLC

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

#include "neuropod/multiprocess/shm/raw_shm_block_allocator.hh"

#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/memory_utils.hh"

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <iostream>
#include <mutex>
#include <unordered_map>

namespace neuropod
{

namespace
{

namespace ipc = boost::interprocess;

// A struct stored in shared memory that contains the data and a cross process refcount
struct __attribute__((__packed__)) RawSHMBlockInternal
{
    // This mutex is used to synchronize operations on the refcount below
    ipc::interprocess_mutex mutex;

    // A reference count for this block
    size_t refcount = 0;

    // The data in this block
    uint8_t data[];

    // Delete the copy and move constructors
    // (if the copy constructors are deleted, no move constructors will be autogenerated)
    RawSHMBlockInternal()                            = default;
    RawSHMBlockInternal(const RawSHMBlockInternal &) = delete;
    RawSHMBlockInternal &operator=(const RawSHMBlockInternal &) = delete;
};

// A utility to get an shm key from a UUID
std::string get_key_from_uuid(const boost::uuids::uuid &uuid)
{
    return "neuropod." + boost::uuids::to_string(uuid);
}

// Used to generate UUIDs for blocks of memory
// `thread_local` so we can avoid locking
thread_local boost::uuids::random_generator uuid_generator;

// A unique handle for a Raw SHM block
// This is currently just a UUID
struct __attribute__((__packed__)) RawSHMHandleInternal
{
    boost::uuids::uuid uuid;
};

// Make sure the size of the handle struct matches the size of the user facing version
static_assert(sizeof(RawSHMHandleInternal) == std::tuple_size<RawSHMHandle>::value,
              "The size of RawSHMHandleInternal must match the size of RawSHMHandle");

// Controls a block of shared memory
class RawSHMBlock
{
private:
    std::unique_ptr<ipc::shared_memory_object> shm_;
    std::unique_ptr<ipc::mapped_region>        region_;

    // A pointer to the struct in shared memory
    RawSHMBlockInternal *block_ = nullptr;

    // The block's UUID
    boost::uuids::uuid uuid_;

public:
    // Allocate a new block of shared memory
    explicit RawSHMBlock(size_t size_bytes)
        // Generate a uuid
        : uuid_(uuid_generator())
    {
        // Create a block of shared memory
        shm_ = stdx::make_unique<ipc::shared_memory_object>(
            ipc::create_only, get_key_from_uuid(uuid_).c_str(), ipc::read_write);

        // Set the size
        shm_->truncate(sizeof(RawSHMBlockInternal) + size_bytes);

        // Map into memory
        region_ = stdx::make_unique<ipc::mapped_region>(*shm_, ipc::read_write);

        // Get a pointer to the struct and initialize it
        block_ = new (region_->get_address()) RawSHMBlockInternal;

        // Increment the refcount
        // Note: we don't need to lock the mutex here because we are the only ones
        // with an active reference to this block
        block_->refcount++;
    }

    // Load an existing block of shared memory from a handle
    explicit RawSHMBlock(const RawSHMHandleInternal *handle)
        // Extract the UUID
        : uuid_(handle->uuid)
    {
        // Get the shm_key
        const auto shm_key = get_key_from_uuid(uuid_);

        // Load a chunk of shared memory
        shm_ = stdx::make_unique<ipc::shared_memory_object>(ipc::open_only, shm_key.c_str(), ipc::read_write);

        // Map into memory
        region_ = stdx::make_unique<ipc::mapped_region>(*shm_, ipc::read_write);

        // Get a pointer to the struct
        block_ = static_cast<RawSHMBlockInternal *>(region_->get_address());

        // Lock the mutex
        ipc::scoped_lock<ipc::interprocess_mutex> lock(block_->mutex);

        // Sanity check
        if (block_->refcount == 0)
        {
            // This means that the other process isn't keeping references to data long enough for this
            // process to load the data.
            // This can lead to some hard to debug race conditions so we always throw an error.
            NEUROPOD_ERROR("Tried getting a pointer to an existing chunk of memory that has a refcount of zero: {}",
                           uuid_);
        }

        // Increment the refcount
        block_->refcount++;
    }

    // Delete copy constructors
    RawSHMBlock(const RawSHMBlock &) = delete;
    RawSHMBlock &operator=(const RawSHMBlock &) = delete;

    ~RawSHMBlock()
    {
        // Lock the mutex
        ipc::scoped_lock<ipc::interprocess_mutex> lock(block_->mutex);

        // Decrement the refcount
        block_->refcount--;

        if (block_->refcount == 0)
        {
            // This block is unused and we're responsible for deleting it

            // Get the shm_key
            const auto shm_key = get_key_from_uuid(uuid_);

            // Unlock the scoped lock before we actually delete
            // This is safe because we're the only one with a reference to this block
            lock.unlock();

            // Unmap memory
            region_ = nullptr;
            shm_    = nullptr;

            // Free the shared memory
            if (!ipc::shared_memory_object::remove(shm_key.c_str()))
            {
                // We shouldn't throw errors from the destructor so let's just
                // log instead
                std::cerr << "Error freeing shared memory with key " << shm_key;
            }
        }
    }

    // Get a pointer to the data stored in shared memory
    void *get_data() { return block_->data; }

    RawSHMHandleInternal get_handle() const { return {uuid_}; }
};

} // namespace

RawSHMBlockAllocator::RawSHMBlockAllocator() = default;

RawSHMBlockAllocator::~RawSHMBlockAllocator() = default;

std::shared_ptr<void> RawSHMBlockAllocator::allocate_shm(size_t size_bytes, RawSHMHandle &handle)
{
    // Create a block of the requested size
    auto block = std::make_shared<RawSHMBlock>(size_bytes);

    // Return the handle of this block to the caller
    auto block_handle = block->get_handle();
    memcpy(handle.data(), &block_handle, sizeof(block_handle));

    // Create a shared pointer to the underlying data with a custom deleter
    // that keeps the block alive
    return std::shared_ptr<void>(block->get_data(), [block](void *unused) {});
}

std::shared_ptr<void> RawSHMBlockAllocator::load_shm(const RawSHMHandle &handle)
{
    // Load an existing block of shared memory given a handle
    auto block = std::make_shared<RawSHMBlock>(reinterpret_cast<const RawSHMHandleInternal *>(handle.data()));

    // Create a shared pointer to the underlying data with a custom deleter
    // that keeps the block alive
    return std::shared_ptr<void>(block->get_data(), [block](void *unused) {});
}

} // namespace neuropod
