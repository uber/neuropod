//
// Uber, Inc. (c) 2019
//

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "neuropods/multiprocess/shm_allocator.hh"
#include "timing_utils.hh"

TEST(test_shm_allocator, memcpy)
{
    const uint8_t    some_image_data[1200 * 1920 * 3] = {0};
    constexpr size_t num_bytes                        = 1200 * 1920 * 3 * sizeof(uint8_t);

    neuropods::SHMAllocator allocator;

    auto shm_time_force_new = time_lambda<std::chrono::microseconds>(100, 500, [&allocator, &some_image_data]() {
        // Force a new allocation
        allocator.free_unused_shm_blocks();

        // Allocate some memory
        neuropods::SHMBlockID block_id;
        auto                  data = allocator.allocate_shm(num_bytes, block_id);

        // Copy in data
        memcpy(data.get(), some_image_data, num_bytes);
    });

    auto shm_time = time_lambda<std::chrono::microseconds>(100, 1000, [&allocator, &some_image_data]() {
        // Allocate some memory
        // This should reuse bocks of memory from previous allocations
        neuropods::SHMBlockID block_id;
        auto                  data = allocator.allocate_shm(num_bytes, block_id);

        // Copy in data
        memcpy(data.get(), some_image_data, num_bytes);
    });

    auto malloc_time = time_lambda<std::chrono::microseconds>(100, 1000, [&some_image_data]() {
        // Allocate some memory
        auto data = malloc(num_bytes);

        // Copy in data
        memcpy(data, some_image_data, num_bytes);

        // Make sure the compiler doesn't optimize this out
        benchmark::DoNotOptimize(data);

        // Free it
        free(data);
    });

    std::cout << "SHM time (microseconds, no reuse): " << shm_time_force_new << std::endl;
    std::cout << "SHM time (microseconds, w/ reuse): " << shm_time << std::endl;
    std::cout << "malloc time microseconds: " << malloc_time << std::endl;

    // Note: we're being generous to avoid flakiness on CI
    // If we're reusing blocks of memory, shm_time should be roughly the same
    // as malloc_time
    EXPECT_LE(shm_time, malloc_time * 5);

    // Allocating every cycle should be much slower than reusing blocks
    EXPECT_LE(shm_time * 5, shm_time_force_new);
}

TEST(test_shm_allocator, simple)
{
    neuropods::SHMAllocator allocator;
    for (uint8_t i = 0; i < 16; i++)
    {
        const uint8_t    some_image_data[1200 * 1920 * 3] = {i};
        constexpr size_t num_bytes                        = 1200 * 1920 * 3 * sizeof(uint8_t);

        // Allocate some memory and copy in data
        neuropods::SHMBlockID block_id;
        auto                  data = allocator.allocate_shm(num_bytes, block_id);
        memcpy(data.get(), some_image_data, num_bytes);

        // Load the block of memory and ensure the data
        // is what we expect
        auto loaded = allocator.load_shm(block_id);
        EXPECT_EQ(memcmp(loaded.get(), some_image_data, num_bytes), 0);
    }
}

TEST(test_shm_allocator, out_of_scope)
{
    neuropods::SHMBlockID block_id;

    // Allocate some shared memory and let everything go out of scope
    {
        neuropods::SHMAllocator allocator;
        auto                    data = allocator.allocate_shm(1024, block_id);
    }

    // Try loading the block we previously allocated
    {
        neuropods::SHMAllocator allocator;
        EXPECT_ANY_THROW(allocator.load_shm(block_id));
    }
}

TEST(test_shm_allocator, stale)
{
    neuropods::SHMBlockID   block_id;
    neuropods::SHMAllocator allocator;

    // Allocate some shared memory and let it go out of scope
    {
        auto data = allocator.allocate_shm(1024, block_id);
    }

    // This allocation should reuse the previously allocated block of memory
    neuropods::SHMBlockID other_id;
    auto                  data = allocator.allocate_shm(1024, other_id);

    // Try loading a block with the original ID (which is now stale)
    // This should throw an error because the block of memory has been reused
    EXPECT_ANY_THROW(allocator.load_shm(block_id));
}
