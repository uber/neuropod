//
// Uber, Inc. (c) 2019
//

#include "benchmark/benchmark.h"
#include "neuropods/backends/test_backend/test_neuropod_backend.hh"
#include "neuropods/neuropods.hh"

static void benchmark_raw(benchmark::State &state)
{
    neuropods::TestNeuropodBackend backend;

    auto allocator = backend.get_tensor_allocator();
    auto tensor = allocator->allocate_tensor<float>({3, 5});

    auto data_ptr = tensor->get_raw_data_ptr();

    for (auto _ : state)
    {
        float *curr = data_ptr;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                curr[i * 5 + j] = i * 5 + j;
            }
        }
    }
}
BENCHMARK(benchmark_raw);

static void benchmark_accessor(benchmark::State &state)
{
    neuropods::TestNeuropodBackend backend;

    auto allocator = backend.get_tensor_allocator();
    auto tensor = allocator->allocate_tensor<float>({3, 5});

    auto accessor = tensor->accessor<2>();

    for (auto _ : state)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                accessor[i][j] = i * 5 + j;
            }
        }
    }
}
BENCHMARK(benchmark_accessor);

// Run the benchmark
BENCHMARK_MAIN();
