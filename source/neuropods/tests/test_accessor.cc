//
// Uber, Inc. (c) 2019
//

#include "timing_utils.hh"
#include "gtest/gtest.h"
#include "neuropods/neuropods.hh"
#include "neuropods/backends/test_backend/test_neuropod_backend.hh"

TEST(test_accessor_timing, test_accessor_timing)
{
    // Test that the config is valid
    neuropods::TestNeuropodBackend backend;
    auto allocator = backend.get_tensor_allocator();

    auto tensor1 = allocator->allocate_tensor<float>({3, 5});
    auto tensor2 = allocator->allocate_tensor<float>({3, 5});

    auto accessor = tensor1->accessor<2>();
    auto data_ptr = tensor2->get_raw_data_ptr();

    auto native_time = time_lambda<std::chrono::nanoseconds>(100, 1000, [data_ptr]() {
        float * curr = data_ptr;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                curr[i * 5 + j] = i * 5 + j;
            }
        }
    });

    auto accessor_time = time_lambda<std::chrono::nanoseconds>(100, 1000, [&accessor]() {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                accessor[i][j] = i * 5 + j;
            }
        }
    });

    // Make sure that tensor 1 and tensor 2 are equal
    EXPECT_EQ(memcmp(tensor1->get_raw_data_ptr(), tensor2->get_raw_data_ptr(), tensor1->get_num_elements() * sizeof(float)), 0);

    std::cout << "Native loop time (nanoseconds): " << native_time << ". Accessor time (nanoseconds): " << accessor_time << std::endl;

    // Note: we're being generous to avoid flakiness on CI
    EXPECT_LE(accessor_time, native_time * 5);
}
