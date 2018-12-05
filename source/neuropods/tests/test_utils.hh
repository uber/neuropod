//
// Uber, Inc. (c) 2018
//

#pragma once

#include <algorithm>
#include <string>

#include "gtest/gtest.h"

#include "neuropods/neuropods.hh"

void test_addition_model(const std::string &neuropod_path, const std::string &backend, bool do_fail)
{
    // Some sample input data
    std::vector<int64_t> shape = {2, 2};

    const float x_data[] = {1, 2, 3, 4};
    float       y_data[] = {7, 8, 9, 10};
    float       target[] = {8, 10, 12, 14};

    if (do_fail)
    {
        target[0] += 1;
    }

    // Load the neuropod
    neuropods::Neuropod neuropod(neuropod_path, backend);

    // Get an input builder and add some data
    auto input_builder = neuropod.get_input_builder();
    auto input_data    = input_builder->add_tensor("x", x_data, 4, shape).add_tensor("y", y_data, 4, shape).build();

    // Run inference
    const auto output_data = neuropod.infer(input_data);

    // Get the data in the output tensor
    const std::vector<float>   out_vector = output_data->get_data_as_vector<float>("out");
    const std::vector<int64_t> out_shape  = output_data->get_shape("out");

    // Check that the output data matches
    EXPECT_EQ(out_vector.size(), 4);

    if (do_fail)
    {
        EXPECT_FALSE(std::equal(out_vector.begin(), out_vector.end(), target));
    }
    else
    {
        EXPECT_TRUE(std::equal(out_vector.begin(), out_vector.end(), target));
    }

    // Check that the shape matches
    EXPECT_TRUE(out_shape == shape);
}

void test_addition_model(const std::string &neuropod_path, const std::string &backend)
{
    // Tests that the output matches the target
    test_addition_model(neuropod_path, backend, true);

    // Output shouldn't match the target
    test_addition_model(neuropod_path, backend, false);
}
