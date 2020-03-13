//
// Uber, Inc. (c) 2020
//

#pragma once

#include <array>
#include <memory>

namespace neuropod
{

// The handle is just 16 opaque bytes (from the perspective of users of this allocator)
using RawSHMHandle = std::array<char, 16>;

// Allocate shared memory blocks of a specific size or load shared memory blocks given a handle
//
// This allocator shouldn't be used directly.
// It is a building block for a performant/usable allocator and does not include many optimizations.
// See `SHMAllocator` for an allocator that includes these optimizations
//
// Note: the methods below are all threadsafe
class RawSHMBlockAllocator
{
public:
    RawSHMBlockAllocator();
    ~RawSHMBlockAllocator();

    // Allocate a block of shared memory of a specific size
    std::shared_ptr<void> allocate_shm(size_t size_bytes, RawSHMHandle &handle);

    // Load an existing block of shared memory given a handle
    std::shared_ptr<void> load_shm(const RawSHMHandle &handle);
};

} // namespace neuropod
