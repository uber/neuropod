//
// Uber, Inc. (c) 2018
//

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "neuropods/neuropods.hh"

void test_addition_model(neuropods::Neuropod &neuropod)
{
    // Check the input and output tensor specs
    auto input_specs  = neuropod.get_inputs();
    auto output_specs = neuropod.get_outputs();
    EXPECT_EQ(input_specs.at(0).name, "x");
    EXPECT_EQ(input_specs.at(0).type, neuropods::FLOAT_TENSOR);

    EXPECT_EQ(input_specs.at(1).name, "y");
    EXPECT_EQ(input_specs.at(1).type, neuropods::FLOAT_TENSOR);

    EXPECT_EQ(output_specs.at(0).name, "out");
    EXPECT_EQ(output_specs.at(0).type, neuropods::FLOAT_TENSOR);

    // Some sample input data
    std::vector<int64_t> shape = {2, 2};

    const float x_data[] = {1, 2, 3, 4};
    const float y_data[] = {7, 8, 9, 10};
    const float target[] = {8, 10, 12, 14};

    // Get an input builder and add some data
    auto input_builder = neuropod.get_input_builder();
    auto input_data    = input_builder->add_tensor("x", x_data, 4, shape).add_tensor("y", y_data, 4, shape).build();

    // Run inference
    const auto output_data = neuropod.infer(input_data);

    // Get the data in the output tensor
    const std::vector<float>   out_vector = output_data->find_or_throw("out")
                                                       ->as_typed_tensor<float>()
                                                       ->get_data_as_vector();

    const std::vector<int64_t> out_shape  = output_data->find_or_throw("out")->get_dims();

    // Check that the output data matches
    EXPECT_EQ(out_vector.size(), 4);
    EXPECT_TRUE(std::equal(out_vector.begin(), out_vector.end(), target));

    // Check that the shape matches
    EXPECT_TRUE(out_shape == shape);
}

void test_addition_model(const std::string &neuropod_path, const std::string &backend)
{
    // Load the neuropod
    neuropods::Neuropod neuropod(neuropod_path, backend);
    test_addition_model(neuropod);
}

void test_addition_model(const std::string &neuropod_path)
{
    // Load the neuropod
    neuropods::Neuropod neuropod(neuropod_path);
    test_addition_model(neuropod);
}


void test_strings_model(neuropods::Neuropod &neuropod)
{
    // Tests a model that concatenates string tensors
    // Some sample input data
    std::vector<int64_t> shape = {3};

    const std::vector<std::string> x_data = {"apple", "banana", "carrot"};
    const std::vector<std::string> y_data = {"sauce", "pudding", "cake"};
    std::vector<std::string>       target = {"apple sauce", "banana pudding", "carrot cake"};

    // Get an input builder and add some data
    auto input_builder = neuropod.get_input_builder();
    auto input_data    = input_builder->add_tensor("x", x_data, shape).add_tensor("y", y_data, shape).build();

    // Run inference
    const auto output_data = neuropod.infer(input_data);

    // Get the data in the output tensor
    const std::vector<std::string> out_vector = output_data->find_or_throw("out")
                                                           ->as_typed_tensor<std::string>()
                                                           ->get_data_as_vector();

    const std::vector<int64_t>     out_shape  = output_data->find_or_throw("out")->get_dims();

    // Check that the output data matches
    EXPECT_EQ(out_vector.size(), 3);
    EXPECT_TRUE(out_vector == target);

    // Check that the shape matches
    EXPECT_TRUE(out_shape == shape);
}

void test_strings_model(const std::string &neuropod_path, const std::string &backend)
{
    // Load the neuropod
    neuropods::Neuropod neuropod(neuropod_path, backend);
    test_strings_model(neuropod);
}

void test_strings_model(const std::string &neuropod_path)
{
    // Load the neuropod
    neuropods::Neuropod neuropod(neuropod_path);
    test_strings_model(neuropod);
}
