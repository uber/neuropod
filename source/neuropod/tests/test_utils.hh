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

#pragma once

#include "gtest/gtest.h"
#include "neuropod/neuropod.hh"

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

#include <stdlib.h>

void test_addition_model(neuropod::Neuropod &neuropod, bool copy_mem)
{
    std::atomic_int free_counter{0};
    {
        // Check name and platform.
        EXPECT_EQ(neuropod.get_name(), "addition_model");
        const auto &p = neuropod.get_platform();
        EXPECT_TRUE(p == "tensorflow" || p == "python" || p == "torchscript");

        // Check the input and output tensor specs
        auto input_specs  = neuropod.get_inputs();
        auto output_specs = neuropod.get_outputs();
        EXPECT_EQ(input_specs.at(0).name, "x");
        EXPECT_EQ(input_specs.at(0).type, neuropod::FLOAT_TENSOR);

        EXPECT_EQ(input_specs.at(1).name, "y");
        EXPECT_EQ(input_specs.at(1).type, neuropod::FLOAT_TENSOR);

        EXPECT_EQ(output_specs.at(0).name, "out");
        EXPECT_EQ(output_specs.at(0).type, neuropod::FLOAT_TENSOR);

        // Some sample input data
        std::vector<int64_t> shape    = {2, 2};
        const float          x_data[] = {1, 2, 3, 4};
        const float          y_data[] = {7, 8, 9, 10};
        const float          target[] = {8, 10, 12, 14};

        neuropod::NeuropodValueMap input_data;

        if (copy_mem)
        {
            // Allocate the tensors
            auto x_ten = neuropod.allocate_tensor<float>(shape);
            auto y_ten = neuropod.allocate_tensor<float>(shape);

            // Copy the input data
            x_ten->copy_from(x_data, 4);
            y_ten->copy_from(y_data, 4);

            // Add it to the inputs for inference
            input_data["x"] = x_ten;
            input_data["y"] = y_ten;
        }
        else
        {
            // 64 byte aligned input data
            float *x_data_aligned, *y_data_aligned;
            EXPECT_EQ(0, posix_memalign((void **) &x_data_aligned, 64, 64));
            EXPECT_EQ(0, posix_memalign((void **) &y_data_aligned, 64, 64));

            // Set the data
            std::copy(x_data, x_data + 4, x_data_aligned);
            std::copy(y_data, y_data + 4, y_data_aligned);

            // Set up a deleter to free the memory
            auto deleter = [&](void *data) {
                free(data);
                free_counter++;
            };

            // Create tensors by wrapping existing data
            input_data["x"] = neuropod.tensor_from_memory(shape, const_cast<float *>(x_data_aligned), deleter);
            input_data["y"] = neuropod.tensor_from_memory(shape, const_cast<float *>(y_data_aligned), deleter);
        }

        // Run inference
        const auto output_data = neuropod.infer(input_data);

        // Get the data in the output tensor
        const std::vector<float> out_vector = output_data->at("out")->as_typed_tensor<float>()->get_data_as_vector();

        const std::vector<int64_t> out_shape = output_data->at("out")->as_tensor()->get_dims();

        // Check that the output data matches
        EXPECT_EQ(out_vector.size(), 4);
        EXPECT_TRUE(std::equal(out_vector.begin(), out_vector.end(), target));

        // Check that the shape matches
        EXPECT_TRUE(out_shape == shape);
    }

    if (!copy_mem)
    {
        // Make sure we ran the deleter
        EXPECT_EQ(free_counter, 2);
    }
}

void test_addition_model(neuropod::Neuropod &neuropod)
{
    // Run the test with and without copying the input data
    test_addition_model(neuropod, true);
    test_addition_model(neuropod, false);
}

void test_addition_model(const std::string &neuropod_path)
{
    // Load the neuropod
    neuropod::RuntimeOptions opts;
    opts.load_model_at_construction = false;
    neuropod::Neuropod neuropod(neuropod_path, opts);

    // Should fail because we haven't loaded the model yet
    EXPECT_ANY_THROW(test_addition_model(neuropod));

    // Load the model and try again
    neuropod.load_model();
    test_addition_model(neuropod);
}

void test_addition_model_ope(const std::string &neuropod_path)
{
    // Load the neuropod
    neuropod::RuntimeOptions opts;
    opts.use_ope = true;
    neuropod::Neuropod neuropod(neuropod_path, opts);
    test_addition_model(neuropod);
}

void test_strings_model(neuropod::Neuropod &neuropod)
{
    // Tests a model that concatenates string tensors
    // Some sample input data
    std::vector<int64_t> shape = {3};

    const std::vector<std::string> x_data = {"apple", "banana", "carrot"};
    const std::vector<std::string> y_data = {"sauce", "pudding", "cake"};
    std::vector<std::string>       target = {"apple sauce", "banana pudding", "carrot cake"};

    // Allocate tensors
    auto x_ten = neuropod.allocate_tensor<std::string>(shape);
    auto y_ten = neuropod.allocate_tensor<std::string>(shape);

    // Set the data
    x_ten->copy_from(x_data);

    // Test another code path
    auto y_accessor = y_ten->accessor<1>();
    for (int i = 0; i < y_data.size(); i++)
    {
        y_accessor[i] = y_data.at(i);
    }

    // Run inference
    // Requesting the "out" tensor here isn't strictly necessary, but is used to test functionality
    const auto output_data = neuropod.infer({{"x", x_ten}, {"y", y_ten}}, {"out"});

    // Get the data in the output tensor
    const std::vector<std::string> out_vector =
        output_data->at("out")->as_typed_tensor<std::string>()->get_data_as_vector();

    const std::vector<int64_t> out_shape = output_data->at("out")->as_tensor()->get_dims();

    // Check that the output data matches
    EXPECT_EQ(out_vector.size(), 3);
    EXPECT_TRUE(out_vector == target);

    // Check that the shape matches
    EXPECT_TRUE(out_shape == shape);
}

void test_strings_model(const std::string &neuropod_path)
{
    // Load the neuropod
    neuropod::RuntimeOptions opts;
    opts.load_model_at_construction = false;
    neuropod::Neuropod neuropod(neuropod_path, opts);

    // Should fail because we haven't loaded the model yet
    EXPECT_ANY_THROW(test_strings_model(neuropod));

    // Load the model and try again
    neuropod.load_model();
    test_strings_model(neuropod);
}

void test_strings_model_ope(const std::string &neuropod_path)
{
    // Load the neuropod
    neuropod::RuntimeOptions opts;
    opts.use_ope = true;
    neuropod::Neuropod neuropod(neuropod_path, opts);
    test_strings_model(neuropod);
}
