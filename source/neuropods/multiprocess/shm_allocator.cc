//
// Uber, Inc. (c) 2019
//

#include "neuropods/multiprocess/shm_allocator.hh"

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <unordered_map>
#include <mutex>

#include "neuropods/internal/error_utils.hh"
#include "neuropods/internal/memory_utils.hh"

namespace neuropods
{

namespace
{

namespace ipc = boost::interprocess;

// A struct stored in shared memory that contains the data and a cross process refcount
struct __attribute__((__packed__)) shm_block
{
    // We need to keep track of a reference count across both processes
    ipc::interprocess_mutex mutex;
    size_t refcount = 0;

    // The data in this block
    uint8_t data[];
};

// Data that does not need to be shared across processes
struct shm_wrapper
{
  std::unique_ptr<ipc::shared_memory_object> shm;
  std::unique_ptr<ipc::mapped_region> region;

  // A pointer to the struct in shared memory
  shm_block *block;

  // The block's UUID
  boost::uuids::uuid uuid;
};

// A pool of objects that the current process has allocated
// which are currently not referenced in this process
std::unordered_multimap<size_t, std::shared_ptr<shm_wrapper>> unused_pool;
std::mutex unused_pool_mutex;

// Used to generate UUIDs for blocks of memory
boost::uuids::random_generator uuid_generator;
std::mutex                     uuid_mutex_;

std::string get_key_from_uuid(const boost::uuids::uuid &uuid)
{
    return "neuropod." + boost::uuids::to_string(uuid);
}

// Allocate a new block of shared memory
std::shared_ptr<shm_wrapper> allocate_new_block(size_t size_bytes)
{
    auto out = std::make_shared<shm_wrapper>();
    // Generate a UUID
    {
        std::lock_guard<std::mutex> lock(uuid_mutex_);
        out->uuid = uuid_generator();
    }

    // Create a block of shared memory
    out->shm = stdx::make_unique<ipc::shared_memory_object>(ipc::create_only, get_key_from_uuid(out->uuid).c_str(), ipc::read_write);

    // Set the size
    out->shm->truncate(sizeof(shm_block) + size_bytes);

    // Map into memory
    out->region = stdx::make_unique<ipc::mapped_region>(*out->shm, ipc::read_write);

    // Get a pointer to the struct and initialize it
    out->block = new (out->region->get_address()) shm_block;

    // Increment the refcount
    // Note: we don't need to lock the mutex here because this is a brand new instance
    // and we are the only ones with a reference to it
    out->block->refcount++;

    return out;
}

// Load an existing block of shared memory based on a UUID
std::shared_ptr<shm_wrapper> load_block_by_uuid(const boost::uuids::uuid &uuid)
{
    auto out = std::make_shared<shm_wrapper>();

    // Load a chunk of shared memory
    out->shm = stdx::make_unique<ipc::shared_memory_object>(ipc::open_only, get_key_from_uuid(uuid).c_str(), ipc::read_write);

    // Map into memory
    out->region = stdx::make_unique<ipc::mapped_region>(*out->shm, ipc::read_write);

    // Get a pointer to the struct
    out->block = static_cast<shm_block *>(out->region->get_address());

    // Set the UUID
    out->uuid = uuid;

    // Increment the refcount
    ipc::scoped_lock<ipc::interprocess_mutex> lock(out->block->mutex);

    if (out->block->refcount == 0)
    {
        // This means that the other process isn't keeping references to data long enough for this
        // process to load the data.
        // This can lead to some hard to debug race conditions so we always throw an error.
        NEUROPOD_ERROR("Tried getting a pointer to an existing chunk of memory that has a refcount of zero");
    }

    out->block->refcount++;

    return out;
}

// Look for an unused block that we have previously allocated that is of size
// `size_bytes`. Return the block if found otherwise return nullptr
std::shared_ptr<shm_wrapper> maybe_get_existing_block(size_t size_bytes)
{
  std::lock_guard<std::mutex> lock(unused_pool_mutex);
  auto range = unused_pool.equal_range(size_bytes);
  for (auto it = range.first; it != range.second; it++)
  {
      auto wrapper = it->second;

      // Lock the refcount mutex
      ipc::scoped_lock<ipc::interprocess_mutex> lock(wrapper->block->mutex);
      if (wrapper->block->refcount == 0)
      {
        // This block is unused!
        // Since the refcount is zero, that means no other process has a reference to it
        // and the only place it is available is in this pool (which we have a lock on)
        // Therefore, it is safe to use

        // "Claim" it by incrementing the refcount
        wrapper->block->refcount++;

        // Remove it from the unused pool
        unused_pool.erase(it);

        // Return the wrapper
        return wrapper;
      }
  }

  return nullptr;
}

} // namespace

// Get a block of shared memory of a specific size
std::shared_ptr<void> get_shm(size_t size_bytes, boost::uuids::uuid &uuid)
{
    // Try to get an existing unused block of the requested size
    auto wrapper = maybe_get_existing_block(size_bytes);

    // Othewise allocate a new one
    if (!wrapper)
    {
        wrapper = allocate_new_block(size_bytes);
    }

    // Create a shared pointer to the underlying data with a custom deleter
    // that drops the cross process refcount and adds to the unused pool
    std::shared_ptr<void> out(wrapper->block->data, [wrapper, size_bytes](void * unused) {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(wrapper->block->mutex);
        wrapper->block->refcount--;

        // This tensor was created by the current process and it is unused in the current process
        // Add it to our unused pool to potentially be reused
        std::lock_guard<std::mutex> pool_lock(unused_pool_mutex);
        std::pair<size_t, std::shared_ptr<shm_wrapper>> item(size_bytes, wrapper);

        unused_pool.insert(item);
    });

    // Return the UUID of this block as well
    uuid = wrapper->uuid;

    return out;
}

// Load an existing block of shared memory by uuid
std::shared_ptr<void> load_shm(const boost::uuids::uuid &uuid)
{
    auto wrapper = load_block_by_uuid(uuid);

    // Create a shared pointer to the underlying data with a custom deleter
    // that drops the cross process refcount
    std::shared_ptr<void> out(wrapper->block->data, [wrapper](void * unused) {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(wrapper->block->mutex);
        wrapper->block->refcount--;
    });

    return out;
}

// Free all currently unused blocks that were allocated by this process
void free_unused_shm_blocks()
{
    std::lock_guard<std::mutex> pool_lock(unused_pool_mutex);
    for (auto it = unused_pool.begin(); it != unused_pool.end();)
    {
        auto wrapper = it->second;

        // Lock the refcount mutex
        ipc::scoped_lock<ipc::interprocess_mutex> lock(wrapper->block->mutex);
        if (wrapper->block->refcount == 0)
        {
          // This block is unused!
          // Since the refcount is zero, that means no other process has a reference to it
          // and the only place it is available is in this pool (which we have a lock on)
          // Therefore, it is safe to delete

          // Get the shm_key
          auto shm_key = get_key_from_uuid(wrapper->uuid);

          // Unlock the scoped lock before we actually delete
          // This is safe because we're the only one with a reference to this block
          lock.unlock();

          // Unmap memory
          wrapper->region = nullptr;
          wrapper->shm = nullptr;

          // Free the shared memory
          if (!ipc::shared_memory_object::remove(shm_key.c_str()))
          {
              NEUROPOD_ERROR("Error freeing shared memory with key " << shm_key);
          }

          // Remove it from the unused pool
          it = unused_pool.erase(it);
        }
        else
        {
            // Move to the next item
            it++;
        }
    }
}

} // namespace neuropods
