//
// Uber, Inc. (c) 2019
//

#include <memory>

#include <boost/uuid/uuid.hpp>

namespace neuropods
{

// This shared memory allocator is based on a simple hypothesis:
// If a process requests to allocate `N` bytes of shared memory, it is likely to request
// `N` bytes of shared memory again. This is especially likely for repeated inference
// with deep learning models.
//
// Internally, we maintain a pool of blocks of memory that we have allocated in the past.
// If any of those blocks are unused and match the size being requested, we reuse one
// of those blocks instead of allocating new memory.
//
// This is important because reusing previously allocated blocks leads to significantly
// faster memory operations than using newly allocated blocks.
//
// To free all the currently unused blocks, call `free_unused_shm_blocks`. This should
// periodically be called to ensure that unused shared memory is freed.
// Note: the three functions below are all threadsafe


// Get a block of shared memory of a specific size
std::shared_ptr<void> get_shm(size_t size_bytes, boost::uuids::uuid &uuid);

// Load an existing block of shared memory by uuid
std::shared_ptr<void> load_shm(const boost::uuids::uuid &uuid);

// Free all currently unused blocks that were allocated by this process
void free_unused_shm_blocks();

} // namespace neuropods
