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

namespace
{

template <typename T>
bool is_full(neuropod::TypedNeuropodTensor<T> &tensor, T value, size_t num_items)
{
    // Make sure our size is correct
    auto actual_numel = tensor.get_num_elements();
    EXPECT_EQ(actual_numel, num_items);

    // Generate expected data
    T expected_data[num_items];
    std::fill_n(expected_data, num_items, value);

    // Compare
    auto actual_data = tensor.get_raw_data_ptr();
    return memcmp(actual_data, expected_data, num_items * sizeof(uint8_t)) == 0;
}

template <typename T>
bool matches_expected(neuropod::TypedNeuropodTensor<T> &tensor, const std::vector<T> expected)
{
    return tensor.get_data_as_vector() == expected;
}

} // namespace

TEST(test_factories, test_factories)
{
    auto allocator = neuropod::get_generic_tensor_allocator();

    constexpr size_t           num_items = 60;
    const std::vector<int64_t> dims      = {3, 4, 5};

    auto zeros = allocator->zeros<uint16_t>(dims);
    auto ones  = allocator->ones<int32_t>(dims);
    auto full  = allocator->full<double>(dims, 1.23);
    auto randn = allocator->randn<float>(dims);

    EXPECT_TRUE(is_full<uint16_t>(*zeros, 0, num_items));
    EXPECT_TRUE(is_full<int32_t>(*ones, 1, num_items));
    EXPECT_TRUE(is_full<double>(*full, 1.23, num_items));

    // It's super unlikely for every element to be 0
    EXPECT_FALSE(is_full<float>(*randn, 0, num_items));

    auto range1 = allocator->arange<float>(5);
    auto range2 = allocator->arange<float>(2, 6);
    auto range3 = allocator->arange<float>(0, 10, 2);

    // Make sure they're 1D tensors
    EXPECT_EQ(range1->get_dims().size(), 1);
    EXPECT_EQ(range2->get_dims().size(), 1);
    EXPECT_EQ(range3->get_dims().size(), 1);

    // And that they match the expected data
    EXPECT_TRUE(matches_expected(*range1, {0, 1, 2, 3, 4}));
    EXPECT_TRUE(matches_expected(*range2, {2, 3, 4, 5}));
    EXPECT_TRUE(matches_expected(*range3, {0, 2, 4, 6, 8}));

    auto eye1 = allocator->eye<float>(4, 4);
    // clang-format off
    EXPECT_TRUE(matches_expected(*eye1, {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    }));
    // clang-format on

    auto eye2 = allocator->eye<float>(3, 7);
    // clang-format off
    EXPECT_TRUE(matches_expected(*eye2, {
        1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0
    }));
    // clang-format on
}
