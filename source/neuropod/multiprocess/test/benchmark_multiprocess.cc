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
#include "neuropod/neuropod.hh"

namespace
{

struct load_in_process
{
    std::unique_ptr<neuropod::Neuropod> operator()(const std::string &path)
    {
        return neuropod::stdx::make_unique<neuropod::Neuropod>(path);
    }
};

struct load_out_of_process
{
    std::unique_ptr<neuropod::Neuropod> operator()(const std::string &path)
    {
        neuropod::RuntimeOptions opts;
        opts.use_ope = true;
        return neuropod::stdx::make_unique<neuropod::Neuropod>(path, opts);
    }
};

} // namespace

template <typename Loader>
void benchmark_object_detection(benchmark::State &state)
{
    const uint8_t some_image_data[1200 * 1920 * 3] = {0};

    auto neuropod = Loader()("neuropod/tests/test_data/dummy_object_detection/");

    for (auto _ : state)
    {
        neuropod::NeuropodValueMap input_data;

        // Add an input "image"
        auto image_tensor = neuropod->template allocate_tensor<uint8_t>({1200, 1920, 3});
        image_tensor->copy_from(some_image_data, 1200 * 1920 * 3);
        input_data["image"] = image_tensor;

        // Run inference
        const auto output_data = neuropod->infer(input_data);

        // Make sure we don't optimize it out
        benchmark::DoNotOptimize(output_data);
    }
}

BENCHMARK_TEMPLATE(benchmark_object_detection, load_in_process);
BENCHMARK_TEMPLATE(benchmark_object_detection, load_out_of_process);

template <typename Loader>
void benchmark_small_inputs(benchmark::State &state)
{
    const float some_data[10 * 5] = {0};

    auto neuropod = Loader()("neuropod/tests/test_data/dummy_small_input_model/");

    for (auto _ : state)
    {
        neuropod::NeuropodValueMap input_data;

        for (int i = 0; i < 100; i++)
        {
            // Add all the inputs
            auto tensor = neuropod->template allocate_tensor<float>({10, 5});
            tensor->copy_from(some_data, 10 * 5);
            input_data["small_input" + std::to_string(i)] = tensor;
        }

        // Run inference
        const auto output_data = neuropod->infer(input_data);

        // Make sure we don't optimize it out
        benchmark::DoNotOptimize(output_data);
    }
}

BENCHMARK_TEMPLATE(benchmark_small_inputs, load_in_process);
BENCHMARK_TEMPLATE(benchmark_small_inputs, load_out_of_process);
