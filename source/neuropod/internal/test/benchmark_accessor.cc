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
#include "neuropod/core/generic_tensor.hh"
#include "neuropod/neuropod.hh"

static void benchmark_raw(benchmark::State &state)
{
    auto allocator = neuropod::get_generic_tensor_allocator();
    auto tensor    = allocator->allocate_tensor<float>({3, 5});

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
    auto allocator = neuropod::get_generic_tensor_allocator();
    auto tensor    = allocator->allocate_tensor<float>({3, 5});

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
