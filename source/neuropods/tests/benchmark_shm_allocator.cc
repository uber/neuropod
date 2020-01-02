//
// Uber, Inc. (c) 2019
//

// Don't run infer on this file
// NEUROPODS_CI_SKIP_INFER

#include "benchmark/benchmark.h"
#include "neuropods/multiprocess/shm_allocator.hh"

#include <string.h>

namespace
{

// Sample image data
const uint8_t    some_image_data[1200 * 1920 * 3] = {0};
constexpr size_t num_bytes                        = 1200 * 1920 * 3 * sizeof(uint8_t);

} // namespace

static void benchmark_shm_force_new(benchmark::State &state)
{
    neuropod::SHMAllocator allocator;

    for (auto _ : state)
    {
        // Force a new allocation
        allocator.free_unused_shm_blocks();

        // Allocate some memory
        neuropod::SHMBlockID block_id;
        auto                  data = allocator.allocate_shm(num_bytes, block_id);

        // Copy in data
        memcpy(data.get(), some_image_data, num_bytes);
    }
}
BENCHMARK(benchmark_shm_force_new);

static void benchmark_shm(benchmark::State &state)
{
    neuropod::SHMAllocator allocator;

    for (auto _ : state)
    {
        // Allocate some memory
        // This should reuse bocks of memory from previous allocations
        neuropod::SHMBlockID block_id;
        auto                  data = allocator.allocate_shm(num_bytes, block_id);

        // Copy in data
        memcpy(data.get(), some_image_data, num_bytes);
    }
}
BENCHMARK(benchmark_shm);

static void benchmark_malloc(benchmark::State &state)
{
    neuropod::SHMAllocator allocator;

    for (auto _ : state)
    {
        // Allocate some memory
        auto data = malloc(num_bytes);

        // Copy in data
        memcpy(data, some_image_data, num_bytes);

        // Make sure the compiler doesn't optimize this out
        benchmark::DoNotOptimize(data);

        // Free it
        free(data);
    }
}
BENCHMARK(benchmark_malloc);

// Run the benchmark
BENCHMARK_MAIN();
