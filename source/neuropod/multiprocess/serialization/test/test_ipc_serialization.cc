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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "neuropod/multiprocess/ope_load_config.hh"
#include "neuropod/multiprocess/serialization/ipc_serialization.hh"
#include "neuropod/multiprocess/shm_tensor.hh"

namespace
{

template <typename T>
T serialize_deserialize(const T &item)
{
    // Serialize the item
    std::stringstream ss;
    neuropod::ipc_serialize(ss, item);

    // Deserialize the item
    T out;
    neuropod::ipc_deserialize(ss, out);
    return std::move(out);
}

} // namespace

TEST(test_ipc_serialization, vector)
{
    std::vector<std::string> expected = {"some", "vector", "of", "strings"};
    const auto               actual   = serialize_deserialize(expected);

    EXPECT_EQ(expected, actual);
}

TEST(test_ipc_serialization, primitive_type)
{
    uint64_t   expected = 42;
    const auto actual   = serialize_deserialize(expected);

    EXPECT_EQ(expected, actual);
}

TEST(test_ipc_serialization, bool)
{
    {
        bool       expected = true;
        const auto actual   = serialize_deserialize(expected);
        EXPECT_EQ(expected, actual);
    }
    {
        bool       expected = false;
        const auto actual   = serialize_deserialize(expected);
        EXPECT_EQ(expected, actual);
    }
}

TEST(test_ipc_serialization, ope_load_config)
{
    neuropod::ope_load_config expected;
    expected.neuropod_path             = "/some/path";
    expected.default_backend_overrides = {
        {"tensorflow", "1.1.0", "/some/path/to/neuropod_tensorflow_backend.so"},
        {"torchscript", "1.12.0", "/some/path/to/neuropod_torchscrtipt_backend.so"},
    };

    expected.opts.visible_device = neuropod::Device::GPU2;

    const auto actual = serialize_deserialize(expected);

    EXPECT_EQ(expected.neuropod_path, actual.neuropod_path);
    EXPECT_EQ(expected.default_backend_overrides, actual.default_backend_overrides);
    EXPECT_EQ(expected.opts.visible_device, actual.opts.visible_device);
}

TEST(test_ipc_serialization, neuropod_value_map)
{
    // A tensor allocator that allocates tensors in shared memory
    std::unique_ptr<neuropod::NeuropodTensorAllocator> allocator =
        neuropod::stdx::make_unique<neuropod::DefaultTensorAllocator<neuropod::SHMNeuropodTensor>>();

    const std::shared_ptr<neuropod::NeuropodValue> float_tensor_1D =
        allocator->allocate_tensor({3}, neuropod::FLOAT_TENSOR);
    float_tensor_1D->as_typed_tensor<float>()->copy_from({0.0, 0.1, 0.2});

    const std::shared_ptr<neuropod::NeuropodValue> string_tensor_2D =
        allocator->allocate_tensor({2, 2}, neuropod::STRING_TENSOR);
    string_tensor_2D->as_typed_tensor<std::string>()->copy_from({"A", "B", "C", "D"});

    const std::shared_ptr<neuropod::NeuropodValue> int_tensor_2D =
        allocator->allocate_tensor({2, 3}, neuropod::INT32_TENSOR);
    int_tensor_2D->as_typed_tensor<int32_t>()->copy_from({0, 1, 2, 3, 4, 5});

    // Create a map
    neuropod::NeuropodValueMap expected;
    expected["int"]    = int_tensor_2D;
    expected["string"] = string_tensor_2D;
    expected["float"]  = float_tensor_1D;

    const auto actual = serialize_deserialize(expected);

    EXPECT_EQ(*int_tensor_2D, *actual.at("int"));
    EXPECT_EQ(*string_tensor_2D, *actual.at("string"));
    EXPECT_EQ(*float_tensor_1D, *actual.at("float"));
}

TEST(test_ipc_serialization, empty_neuropod_value_map)
{
    neuropod::NeuropodValueMap expected;
    const auto                 actual = serialize_deserialize(expected);

    EXPECT_TRUE(actual.empty());
}

TEST(test_ipc_serialization, large_string)
{
    std::string expected(5000, 'a');
    const auto  actual = serialize_deserialize(expected);
    EXPECT_EQ(expected, actual);
}
