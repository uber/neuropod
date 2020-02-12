//
// Uber, Inc. (c) 2019
//

// Don't run infer on this file
// NEUROPOD_CI_SKIP_INFER

#include "benchmark/benchmark.h"
#include "neuropod/multiprocess/multiprocess.hh"
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
        return neuropod::load_neuropod_in_new_process(path);
    }
};

struct sealed_map_creator
{
    neuropod::SealedNeuropodValueMap operator()(neuropod::Neuropod &neuropod)
    {
        return neuropod::SealedNeuropodValueMap(neuropod);
    }
};

struct regular_map_creator
{
    neuropod::NeuropodValueMap operator()(neuropod::Neuropod &neuropod) { return neuropod::NeuropodValueMap(); }
};

} // namespace

template <typename Loader, typename MapCreator>
void benchmark_object_detection(benchmark::State &state)
{
    const uint8_t some_image_data[1200 * 1920 * 3] = {0};

    auto neuropod = Loader()("neuropod/tests/test_data/dummy_object_detection/");

    for (auto _ : state)
    {
        auto input_data = MapCreator()(*neuropod);

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

BENCHMARK_TEMPLATE(benchmark_object_detection, load_in_process, regular_map_creator);
BENCHMARK_TEMPLATE(benchmark_object_detection, load_in_process, sealed_map_creator);
BENCHMARK_TEMPLATE(benchmark_object_detection, load_out_of_process, regular_map_creator);
BENCHMARK_TEMPLATE(benchmark_object_detection, load_out_of_process, sealed_map_creator);

template <typename Loader, typename MapCreator>
void benchmark_small_inputs(benchmark::State &state)
{
    const float some_data[10 * 5] = {0};

    auto neuropod = Loader()("neuropod/tests/test_data/dummy_small_input_model/");

    for (auto _ : state)
    {
        auto input_data = MapCreator()(*neuropod);

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

BENCHMARK_TEMPLATE(benchmark_small_inputs, load_in_process, regular_map_creator);
BENCHMARK_TEMPLATE(benchmark_small_inputs, load_in_process, sealed_map_creator);
BENCHMARK_TEMPLATE(benchmark_small_inputs, load_out_of_process, regular_map_creator);
BENCHMARK_TEMPLATE(benchmark_small_inputs, load_out_of_process, sealed_map_creator);
