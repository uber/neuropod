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

#include "gtest/gtest.h"
#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/core/generic_tensor.hh"

namespace
{

const std::vector<neuropod::TensorSpec> TEST_SPEC = {
    // (None, 2)
    {"x", {-1, 2}, neuropod::FLOAT_TENSOR},

    // (None, 2)
    {"y", {-1, 2}, neuropod::FLOAT_TENSOR},
};

} // namespace

TEST(test_spec_validation, test_correct_inputs)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    neuropod::NeuropodValueMap inputs;
    inputs["x"] = allocator->allocate_tensor<float>({2, 2});
    inputs["y"] = allocator->allocate_tensor<float>({2, 2});

    // The tensors match the specs so we don't expect an error
    neuropod::validate_tensors_against_specs(inputs, TEST_SPEC);
}

// For now, all tensors are optional
// TEST(test_spec_validation, test_missing_tensor)
// {
//     neuropod::TestNeuropodBackend backend;
//     auto allocator = backend.get_tensor_allocator();

//     neuropod::NeuropodValueMap inputs;
//     inputs["x"] = allocator->allocate_tensor<float>({2, 2});
//     EXPECT_THROW(neuropod::validate_tensors_against_specs(inputs, TEST_SPEC), std::runtime_error);
// }

// For now, all tensors are optional
// TEST(test_spec_validation, test_missing_tensors)
// {
//     neuropod::TestNeuropodBackend backend;
//     auto allocator = backend.get_tensor_allocator();

//     neuropod::NeuropodValueMap inputs;
//     EXPECT_THROW(neuropod::validate_tensors_against_specs(inputs, TEST_SPEC), std::runtime_error);
// }

TEST(test_spec_validation, test_bogus_tensor_name)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    neuropod::NeuropodValueMap inputs;
    inputs["x"]     = allocator->allocate_tensor<float>({2, 2});
    inputs["bogus"] = allocator->allocate_tensor<float>({2, 2});
    EXPECT_THROW(neuropod::validate_tensors_against_specs(inputs, TEST_SPEC), std::runtime_error);
}

TEST(test_spec_validation, test_incorrect_dtype)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    neuropod::NeuropodValueMap inputs;
    inputs["x"] = allocator->allocate_tensor<int32_t>({2, 2});
    inputs["y"] = allocator->allocate_tensor<int32_t>({2, 2});

    // Incorrect dtype (expected float, got int32)
    EXPECT_THROW(neuropod::validate_tensors_against_specs(inputs, TEST_SPEC), std::runtime_error);
}

TEST(test_spec_validation, test_invalid_num_dims)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    neuropod::NeuropodValueMap inputs;
    inputs["x"] = allocator->allocate_tensor<float>({2, 2});
    inputs["y"] = allocator->allocate_tensor<float>({2});

    // "y" only has one dim
    EXPECT_THROW(neuropod::validate_tensors_against_specs(inputs, TEST_SPEC), std::runtime_error);
}

TEST(test_spec_validation, test_invalid_shape)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    neuropod::NeuropodValueMap inputs;
    inputs["x"] = allocator->allocate_tensor<float>({2, 2});
    inputs["y"] = allocator->allocate_tensor<float>({2, 1});

    // Dim 1 of "y" is incorrect
    EXPECT_THROW(neuropod::validate_tensors_against_specs(inputs, TEST_SPEC), std::runtime_error);
}

TEST(test_spec_validation, test_correct_symbol)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    neuropod::NeuropodValueMap inputs;
    inputs["x"] = allocator->allocate_tensor<float>({3, 2});
    inputs["y"] = allocator->allocate_tensor<float>({2, 3});

    const std::vector<neuropod::TensorSpec> SPEC = {
        // ("some_symbol", 2)
        {"x", {-2, 2}, neuropod::FLOAT_TENSOR},

        // (None, "some_symbol")
        {"y", {-1, -2}, neuropod::FLOAT_TENSOR},
    };

    // The tensors matches the specs so we don't expect an error
    neuropod::validate_tensors_against_specs(inputs, SPEC);
}

TEST(test_spec_validation, test_incorrect_symbol)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    neuropod::NeuropodValueMap inputs;
    inputs["x"] = allocator->allocate_tensor<float>({1, 2});
    inputs["y"] = allocator->allocate_tensor<float>({2, 3});

    const std::vector<neuropod::TensorSpec> SPEC = {
        // ("some_symbol", 2)
        {"x", {-2, 2}, neuropod::FLOAT_TENSOR},

        // (None, "some_symbol")
        {"y", {-1, -2}, neuropod::FLOAT_TENSOR},
    };

    // Dim 1 of y should be the same as dim 0 of x (2 != 1)
    EXPECT_THROW(neuropod::validate_tensors_against_specs(inputs, SPEC), std::runtime_error);
}

TEST(test_spec_validation, test_invalid_shape_entry)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    neuropod::NeuropodValueMap inputs;
    inputs["x"] = allocator->allocate_tensor<float>({2, 2});
    inputs["y"] = allocator->allocate_tensor<float>({2, 2});

    const std::vector<neuropod::TensorSpec> SPEC = {
        {"x", {0, 2}, neuropod::FLOAT_TENSOR},

        // (None, 2)
        {"y", {-1, 2}, neuropod::FLOAT_TENSOR},
    };

    // `0` is an invalid dim size in the spec
    EXPECT_THROW(neuropod::validate_tensors_against_specs(inputs, SPEC), std::runtime_error);
}

TEST(test_spec_validation, test_string_tensors)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    neuropod::NeuropodValueMap inputs;
    inputs["x"] = allocator->allocate_tensor<std::string>({1, 3});

    const std::vector<neuropod::TensorSpec> SPEC = {
        {"x", {1, 3}, neuropod::STRING_TENSOR},
    };

    // The tensor matches the spec so we don't expect an error
    neuropod::validate_tensors_against_specs(inputs, SPEC);
}
