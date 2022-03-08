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

// Don't run infer on this file
// NEUROPOD_CI_SKIP_INFER

#include "benchmark/benchmark.h"
#include "neuropod/multiprocess/shm/shm_allocator.hh"

#include <cstring>

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
        auto                 data = allocator.allocate_shm(num_bytes, block_id);

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
        auto                 data = allocator.allocate_shm(num_bytes, block_id);

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
        auto data = malloc(num_bytes); // NOLINT(cppcoreguidelines-no-malloc)

        // Copy in data
        memcpy(data, some_image_data, num_bytes);

        // Make sure the compiler doesn't optimize this out
        benchmark::DoNotOptimize(data);

        // Free it
        free(data); // NOLINT(cppcoreguidelines-no-malloc)
    }
}
BENCHMARK(benchmark_malloc);

// Run the benchmark
BENCHMARK_MAIN();
