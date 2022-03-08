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

#include "gtest/gtest.h"
#include "neuropod/multiprocess/shm/shm_allocator.hh"

TEST(test_shm_allocator, simple)
{
    neuropod::SHMAllocator allocator;
    for (uint8_t i = 0; i < 16; i++)
    {
        const uint8_t    some_image_data[1200 * 1920 * 3] = {i};
        constexpr size_t num_bytes                        = 1200 * 1920 * 3 * sizeof(uint8_t);

        // Allocate some memory and copy in data
        neuropod::SHMBlockID block_id;
        auto                 data = allocator.allocate_shm(num_bytes, block_id);
        memcpy(data.get(), some_image_data, num_bytes);

        // Load the block of memory and ensure the data
        // is what we expect
        auto loaded = allocator.load_shm(block_id);
        EXPECT_EQ(memcmp(loaded.get(), some_image_data, num_bytes), 0);
    }
}

TEST(test_shm_allocator, out_of_scope)
{
    neuropod::SHMBlockID block_id;

    // Allocate some shared memory and let everything go out of scope
    {
        neuropod::SHMAllocator allocator;
        auto                   data = allocator.allocate_shm(1024, block_id);
    }

    // Try loading the block we previously allocated
    {
        neuropod::SHMAllocator allocator;
        EXPECT_ANY_THROW(allocator.load_shm(block_id));
    }
}

TEST(test_shm_allocator, stale)
{
    neuropod::SHMBlockID   block_id;
    neuropod::SHMAllocator allocator;

    // Allocate some shared memory and let it go out of scope
    {
        auto data = allocator.allocate_shm(1024, block_id);
    }

    // This allocation should reuse the previously allocated block of memory
    neuropod::SHMBlockID other_id;
    auto                 data = allocator.allocate_shm(1024, other_id);

    // Try loading a block with the original ID (which is now stale)
    // This should throw an error because the block of memory has been reused
    EXPECT_ANY_THROW(allocator.load_shm(block_id));
}
