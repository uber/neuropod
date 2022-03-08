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
#include "neuropod/serialization/serialization.hh"

void benchmark_serialize_object_detection(benchmark::State &state)
{
    const uint8_t some_image_data[1200 * 1920 * 3] = {0};

    auto allocator = neuropod::get_generic_tensor_allocator();

    for (auto _ : state)
    {
        neuropod::NeuropodValueMap input_data;

        // Add an input "image"
        auto image_tensor = allocator->allocate_tensor<uint8_t>({1200, 1920, 3});
        image_tensor->copy_from(some_image_data, 1200 * 1920 * 3);
        input_data["image"] = image_tensor;

        // Serialize the inputs
        std::stringstream ss;
        neuropod::serialize(ss, input_data);

        // Deserialize the inputs
        auto value = neuropod::deserialize<neuropod::NeuropodValueMap>(ss, *allocator);

        // Make sure we don't optimize it out
        benchmark::DoNotOptimize(value);
    }
}

BENCHMARK(benchmark_serialize_object_detection);

void benchmark_serialize_small_inputs(benchmark::State &state)
{
    const float some_data[10 * 5] = {0};

    auto allocator = neuropod::get_generic_tensor_allocator();

    for (auto _ : state)
    {
        neuropod::NeuropodValueMap input_data;

        for (int i = 0; i < 100; i++)
        {
            // Add all the inputs
            auto tensor = allocator->allocate_tensor<float>({10, 5});
            tensor->copy_from(some_data, 10 * 5);
            input_data["small_input" + std::to_string(i)] = tensor;
        }

        // Serialize the inputs
        std::stringstream ss;
        neuropod::serialize(ss, input_data);

        // Deserialize the inputs
        auto value = neuropod::deserialize<neuropod::NeuropodValueMap>(ss, *allocator);

        // Make sure we don't optimize it out
        benchmark::DoNotOptimize(value);
    }
}

BENCHMARK(benchmark_serialize_small_inputs);
