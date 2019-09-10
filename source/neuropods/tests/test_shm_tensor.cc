//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"
#include "neuropods/multiprocess/shm_tensor.hh"
#include "timing_utils.hh"

TEST(test_shm_tensor, simple)
{
    // A tensor allocator that allocates tensors in shared memory
    std::unique_ptr<neuropods::NeuropodTensorAllocator> allocator =
        neuropods::stdx::make_unique<neuropods::DefaultTensorAllocator<neuropods::SHMNeuropodTensor>>();

    // Store tensors we allocate so they don't go out of scope
    std::vector<std::shared_ptr<neuropods::NeuropodTensor>> items;
    std::vector<neuropods::SHMBlockID>                      block_ids;

    // Sample data
    constexpr size_t           num_items = 1024;
    const std::vector<int64_t> dims      = {2, 4, 8, 16};

    // Allocate some tensors
    for (uint8_t i = 0; i < 16; i++)
    {
        const uint8_t some_data[num_items] = {i};

        // Allocate some memory and copy in data
        auto tensor = allocator->allocate_tensor<uint8_t>(dims);
        tensor->copy_from(some_data, num_items);

        // Store the block ID
        const auto &block_id =
            std::dynamic_pointer_cast<neuropods::NativeDataContainer<neuropods::SHMBlockID>>(tensor)->get_native_data();

        block_ids.emplace_back(block_id);

        // Store the tensor
        items.emplace_back(tensor);
    }

    // Load the tensors and make sure they match what we expect
    for (uint8_t i = 0; i < 16; i++)
    {
        // Load the block of memory and ensure the data
        // is what we expect
        auto tensor = neuropods::tensor_from_id(block_ids.at(i));

        // Make sure dims match
        auto actual_dims = tensor->get_dims();
        EXPECT_EQ(actual_dims, dims);

        // Make sure the data is what we expect
        const uint8_t expected_data[num_items] = {i};
        auto          actual_data              = tensor->as_typed_tensor<uint8_t>()->get_raw_data_ptr();
        EXPECT_EQ(memcmp(actual_data, expected_data, num_items * sizeof(uint8_t)), 0);
    }
}
