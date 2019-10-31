//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"
#include "neuropods/backends/test_backend/test_neuropod_backend.hh"
#include "neuropods/neuropods.hh"

TEST(test_accessor, test_accessor)
{
    // Test that the config is valid
    neuropods::TestNeuropodBackend backend;
    auto                           allocator = backend.get_tensor_allocator();

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
