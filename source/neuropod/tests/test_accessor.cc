//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"
#include "neuropods/backends/test_backend/test_neuropod_backend.hh"
#include "neuropods/neuropods.hh"

TEST(test_accessor, test_accessor)
{
    neuropod::TestNeuropodBackend backend;
    auto                          allocator = backend.get_tensor_allocator();

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
    neuropod::TestNeuropodBackend backend;
    auto                          allocator = backend.get_tensor_allocator();

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
    neuropod::TestNeuropodBackend backend;
    auto                          allocator = backend.get_tensor_allocator();

    auto tensor1 = allocator->allocate_tensor<float>({3, 5});

    // tensor1 has 2 dims, not 3
    EXPECT_THROW(tensor1->accessor<3>(), std::runtime_error);
}
