//
// Uber, Inc. (c) 2019
//

#include <array>
#include <memory>

namespace neuropod
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
// Note: we don't yet optimally handle cases where tensor sizes change every cycle.
// This will be added in the future.
//
// To free all the currently unused blocks, call `free_unused_shm_blocks`. This should
// periodically be called to ensure that unused shared memory is freed.
// Note: the three functions below are all threadsafe

// The block ID is just 24 opaque bytes (from the perspective of users of this allocator)
using SHMBlockID = std::array<char, 24>;

class UnusedPool;
class SHMAllocator
{
private:
    // Keeps track of unused blocks of shared memory so we can reuse them
    std::unique_ptr<UnusedPool> unused_pool_;

public:
    SHMAllocator();
    ~SHMAllocator();

    // Allocate a block of shared memory of a specific size
    // (potentially reusing an unused previously allocated block)
    std::shared_ptr<void> allocate_shm(size_t size_bytes, SHMBlockID &block_id);

    // Load an existing block of shared memory by ID
    std::shared_ptr<void> load_shm(const SHMBlockID &block_id);

    // Free all currently unused blocks that were allocated by this process
    void free_unused_shm_blocks();
};

} // namespace neuropod
