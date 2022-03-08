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
#include "neuropod/core/generic_tensor.hh"
#include "neuropod/neuropod.hh"

TEST(test_accessor, test_accessor)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    auto tensor1 = allocator->allocate_tensor<float>({3, 5});
    auto tensor2 = allocator->allocate_tensor<float>({3, 5});

    auto accessor = tensor1->accessor<2>();
    auto data_ptr = tensor2->get_raw_data_ptr();

    // Manual indexing
    {
        float *curr = data_ptr;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                curr[i * 5 + j] = i * 5 + j;
            }
        }
    };

    // Accessor
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                accessor[i][j] = i * 5 + j;
            }
        }
    };

    // Make sure that tensor 1 and tensor 2 are equal
    EXPECT_EQ(*tensor1, *tensor2);
}

TEST(test_accessor, test_range_loop)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    auto tensor1 = allocator->allocate_tensor<float>({3, 5});

    auto accessor = tensor1->accessor<2>();

    // Set data
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                accessor[i][j] = i * 5 + j;
            }
        }
    };

    // Validate range based loops
    float expected_val = 0;
    for (const auto &row : accessor)
    {
        for (const auto &item : row)
        {
            EXPECT_EQ(item, expected_val++);
        }
    }
    EXPECT_EQ(expected_val, 15);

    // Validate range based loops for const accessors
    const auto &const_accessor = accessor;
    expected_val               = 0;
    for (const auto &row : const_accessor)
    {
        for (const auto &item : row)
        {
            EXPECT_EQ(item, expected_val++);
        }
    }
    EXPECT_EQ(expected_val, 15);
}

TEST(test_accessor, valid_dims)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    auto tensor1 = allocator->allocate_tensor<float>({3, 5});

    // tensor1 has 2 dims, not 3
    EXPECT_THROW(tensor1->accessor<3>(), std::runtime_error);
}

TEST(test_accessor, test_string_read)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    auto tensor = allocator->allocate_tensor<std::string>({3, 5});

    // Manually set data
    {
        std::vector<std::string> to_set;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                to_set.emplace_back(std::to_string(i * 5 + j));
            }
        }

        tensor->copy_from(to_set);
    }

    // Read with an accessor
    const auto accessor = tensor->accessor<2>();
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                EXPECT_EQ(accessor[i][j], std::to_string(i * 5 + j));
            }
        }
    }
}

TEST(test_accessor, test_string_write)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    auto tensor = allocator->allocate_tensor<std::string>({3, 5});

    // Write with an accessor
    const auto               accessor = tensor->accessor<2>();
    std::vector<std::string> expected;
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                auto item      = std::to_string(i * 5 + j);
                accessor[i][j] = item;
                expected.emplace_back(item);
            }
        }
    }

    // Read as a vector
    {
        auto actual = tensor->get_data_as_vector();
        EXPECT_EQ(expected, actual);
    }

    // Get a flat view
    {
        auto actual = tensor->flat();
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), actual.begin()));
    }
}

TEST(test_accessor, view_read)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    auto tensor = allocator->arange<int32_t>(15);

    auto view = tensor->view(3, 5);
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                EXPECT_EQ(view[i][j], i * 5 + j);
            }
        }
    }
}

TEST(test_accessor, view_write)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    auto tensor = allocator->arange<int32_t>(15);

    auto view = tensor->view(3, 5);
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                view[i][j] = i * 5 + j;
            }
        }
    }

    std::vector<int32_t> expected(15);
    std::iota(expected.begin(), expected.end(), 0);

    EXPECT_EQ(expected, tensor->get_data_as_vector());
}

TEST(test_accessor, view_dims)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    auto tensor1 = allocator->allocate_tensor<float>({3, 5});

    // All dims passed to view must be positive
    EXPECT_THROW(tensor1->view(-1, -2), std::runtime_error);

    // Total number of elements must match
    EXPECT_THROW(tensor1->view(4, 2), std::runtime_error);
}
