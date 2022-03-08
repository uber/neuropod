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

#include "neuropod/multiprocess/shm/raw_shm_block_allocator.hh"

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

// The allocator also employs a similar approach for loading blocks:
// If we've loaded a block before, we're likely to load it again.
//
// Internally, we maintain a pool of blocks of memory we've loaded in the past.
// If we are requested to load a block again, we don't need to redo all the work to open
// the shared memory objects.
//
// To free all the currently unused blocks, call `free_unused_shm_blocks`. This should
// periodically be called to ensure that unused shared memory is freed.

// The block ID is just 24 opaque bytes (from the perspective of users of this allocator)
using SHMBlockID = std::array<char, 24>;

// Forward declarations of caches we're using
class AllocationCache;
class LoadCache;

// This allocator builds on top of RawSHMBlockAllocator to implement the optimizations
// described above
//
// Note: the methods below are all threadsafe
class SHMAllocator
{
private:
    RawSHMBlockAllocator allocator_;

    std::unique_ptr<AllocationCache> allocation_cache_;
    std::unique_ptr<LoadCache>       load_cache_;

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

// A shared memory allocator that is used by the WireFormat and by SHMNeuropodTensor
// TODO(vip): Remove global allocator instance
extern SHMAllocator shm_allocator;

} // namespace neuropod
