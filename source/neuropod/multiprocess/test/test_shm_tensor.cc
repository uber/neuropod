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
#include "neuropod/multiprocess/shm_tensor.hh"

TEST(test_shm_tensor, simple)
{
    // A tensor allocator that allocates tensors in shared memory
    std::unique_ptr<neuropod::NeuropodTensorAllocator> allocator =
        neuropod::stdx::make_unique<neuropod::DefaultTensorAllocator<neuropod::SHMNeuropodTensor>>();

    // Store tensors we allocate so they don't go out of scope
    std::vector<std::shared_ptr<neuropod::NeuropodTensor>> items;
    std::vector<neuropod::SHMBlockID>                      block_ids;

    // Sample data
    constexpr size_t           num_items = 1024;
    const std::vector<int64_t> dims      = {2, 4, 8, 16};

    // Allocate some tensors
    for (uint8_t i = 0; i < 16; i++)
    {
        // Allocate a tensor filled with a specific value
        auto tensor = allocator->full<uint8_t>(dims, i);

        // Store the block ID
        const auto &block_id =
            std::dynamic_pointer_cast<neuropod::NativeDataContainer<neuropod::SHMBlockID>>(tensor)->get_native_data();

        block_ids.emplace_back(block_id);

        // Store the tensor
        items.emplace_back(tensor);
    }

    // Load the tensors and make sure they match what we expect
    for (uint8_t i = 0; i < 16; i++)
    {
        // Load the block of memory and ensure the data
        // is what we expect
        auto tensor = neuropod::tensor_from_id(block_ids.at(i));

        // Make sure dims match
        auto actual_dims = tensor->get_dims();
        EXPECT_EQ(actual_dims, dims);

        // Make sure our hardcoded size is correct
        const auto actual_numel = tensor->get_num_elements();
        EXPECT_EQ(actual_numel, num_items);

        // Make sure the data is what we expect
        uint8_t expected_data[num_items];
        std::fill_n(expected_data, num_items, i);

        auto actual_data = tensor->as_typed_tensor<uint8_t>()->get_raw_data_ptr();
        EXPECT_EQ(memcmp(actual_data, expected_data, num_items * sizeof(uint8_t)), 0);
    }
}
