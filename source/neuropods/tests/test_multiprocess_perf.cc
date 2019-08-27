//
// Uber, Inc. (c) 2019
//

#include "neuropods/multiprocess/multiprocess.hh"
#include "neuropods/neuropods.hh"

#include "gtest/gtest.h"

#include <chrono>
#include <numeric>

namespace
{

template<typename T>
float time_lambda_microseconds(size_t warmup, size_t iterations, T fn)
{
    std::vector<size_t> times;

    for (int i = 0; i < warmup + iterations; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        fn();

        auto end = std::chrono::high_resolution_clock::now();

        // Ignore the warmup period
        if (i > warmup)
        {
            times.emplace_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        }
    }

    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}

template <typename T>
void run_in_and_out_of_process(const std::string &neuropod_path, T fn)
{
    auto multiprocess_neuropod = neuropods::load_neuropod_in_new_process(neuropod_path);
    auto multiprocess_time = time_lambda_microseconds(100, 1000, [&multiprocess_neuropod, &fn]() {
        fn(*multiprocess_neuropod);
    });

    neuropods::Neuropod inprocess_neuropod(neuropod_path);
    auto inprocess_time = time_lambda_microseconds(100, 1000, [&inprocess_neuropod, &fn]() {
        fn(inprocess_neuropod);
    });

    std::cout << "Times: in process - " << inprocess_time << " microseconds, out of process - " << multiprocess_time << " microseconds" << std::endl;
}

} // namespace

TEST(test_multiprocess_perf, object_detection)
{
    const uint8_t some_image_data[1200 * 1920 * 3] = {0};
    run_in_and_out_of_process("neuropods/tests/test_data/dummy_object_detection/", [&some_image_data](neuropods::Neuropod &neuropod) {
        neuropods::NeuropodValueMap input_data;

        // Add an input "image"
        auto image_tensor = neuropod.allocate_tensor<uint8_t>({1200, 1920, 3});
        image_tensor->copy_from(some_image_data, 1200 * 1920 * 3);
        input_data["image"] = image_tensor;

        // Run inference
        const auto output_data = neuropod.infer(input_data);
    });
}

TEST(test_multiprocess_perf, small_inputs)
{
    const float some_data[10 * 5] = {0};
    run_in_and_out_of_process("neuropods/tests/test_data/dummy_small_input_model/", [&some_data](neuropods::Neuropod &neuropod) {
        neuropods::NeuropodValueMap input_data;

        for (int i = 0; i < 100; i++)
        {
            // Add all the inputs
            auto tensor = neuropod.allocate_tensor<float>({10, 5});
            tensor->copy_from(some_data, 10 * 5);
            input_data["small_input" + std::to_string(i)] = tensor;
        }

        // Run inference
        const auto output_data = neuropod.infer(input_data);
    });
}
