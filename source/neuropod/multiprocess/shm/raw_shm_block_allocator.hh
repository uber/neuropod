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
