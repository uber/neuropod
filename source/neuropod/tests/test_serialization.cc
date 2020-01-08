//
// Uber, In (c) 2018
//

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "neuropod/backends/test_backend/test_neuropod_backend.hh"
#include "neuropod/serialization/serialization.hh"

std::shared_ptr<neuropod::NeuropodTensor> serialize_deserialize(neuropod::NeuropodTensorAllocator &allocator,
                                                                const neuropod::NeuropodTensor &   tensor)
{
    // Serialize the tensor
    std::stringstream ss;
    neuropod::serialize(ss, tensor);

    // Deserialize the tensor
    auto value = neuropod::deserialize<std::shared_ptr<neuropod::NeuropodValue>>(ss, allocator);
    return std::dynamic_pointer_cast<neuropod::NeuropodTensor>(value);
}

TEST(test_allocate_tensor, serialize_string_tensor)
{
    neuropod::TestNeuropodBackend backend;

    std::vector<std::string> expected_data{"A", "B", "C", "D"};

    auto allocator = backend.get_tensor_allocator();

    // Allocate tensors
    const auto tensor_1D = allocator->allocate_tensor({4}, neuropod::STRING_TENSOR);
    tensor_1D->as_typed_tensor<std::string>()->set(expected_data);

    const auto actual_1D = serialize_deserialize(*allocator, *tensor_1D);
    EXPECT_EQ(*tensor_1D, *actual_1D);

    const auto tensor_2D = allocator->allocate_tensor({2, 2}, neuropod::STRING_TENSOR);
    tensor_2D->as_typed_tensor<std::string>()->set(expected_data);

    const auto actual_2D = serialize_deserialize(*allocator, *tensor_2D);
    EXPECT_EQ(*tensor_2D, *actual_2D);
}

TEST(test_allocate_tensor, serialize_scalar_tensor)
{
    neuropod::TestNeuropodBackend backend;

    auto allocator = backend.get_tensor_allocator();

    const auto tensor_1D = allocator->allocate_tensor({3}, neuropod::FLOAT_TENSOR);
    tensor_1D->as_typed_tensor<float>()->copy_from({0.0, 0.1, 0.2});

    const auto actual_1D = serialize_deserialize(*allocator, *tensor_1D);
    EXPECT_EQ(*tensor_1D, *actual_1D);

    const auto tensor_2D = allocator->allocate_tensor({2, 3}, neuropod::INT32_TENSOR);
    tensor_2D->as_typed_tensor<int32_t>()->copy_from({0, 1, 2, 3, 4, 5});

    const auto actual_2D = serialize_deserialize(*allocator, *tensor_2D);
    EXPECT_EQ(*tensor_2D, *actual_2D);

    const auto tensor_3D = allocator->allocate_tensor({2, 2, 2}, neuropod::INT32_TENSOR);
    tensor_3D->as_typed_tensor<int32_t>()->copy_from({0, 1, 2, 3, 4, 5, 6, 7});

    const auto actual_3D = serialize_deserialize(*allocator, *tensor_3D);
    EXPECT_EQ(*tensor_3D, *actual_3D);
}

TEST(test_allocate_tensor, multiple_tensors_in_a_stream)
{
    neuropod::TestNeuropodBackend backend;

    auto allocator = backend.get_tensor_allocator();

    const auto float_tensor_1D = allocator->allocate_tensor({3}, neuropod::FLOAT_TENSOR);
    float_tensor_1D->as_typed_tensor<float>()->copy_from({0.0, 0.1, 0.2});

    const auto string_tensor_2D = allocator->allocate_tensor({2, 2}, neuropod::STRING_TENSOR);
    string_tensor_2D->as_typed_tensor<std::string>()->set({"A", "B", "C", "D"});

    const auto int_tensor_2D = allocator->allocate_tensor({2, 3}, neuropod::INT32_TENSOR);
    int_tensor_2D->as_typed_tensor<int32_t>()->copy_from({0, 1, 2, 3, 4, 5});

    std::stringstream ss;

    neuropod::serialize(ss, *int_tensor_2D);
    neuropod::serialize(ss, *string_tensor_2D);
    neuropod::serialize(ss, *float_tensor_1D);

    ss.seekg(0, std::ios::beg);

    // Deserialize the tensor
    EXPECT_EQ(*int_tensor_2D, *neuropod::deserialize<std::shared_ptr<neuropod::NeuropodValue>>(ss, *allocator));
    EXPECT_EQ(*string_tensor_2D, *neuropod::deserialize<std::shared_ptr<neuropod::NeuropodValue>>(ss, *allocator));
    EXPECT_EQ(*float_tensor_1D, *neuropod::deserialize<std::shared_ptr<neuropod::NeuropodValue>>(ss, *allocator));
}

TEST(test_allocate_tensor, neuropod_value_map)
{
    neuropod::TestNeuropodBackend backend;

    auto allocator = backend.get_tensor_allocator();

    const std::shared_ptr<neuropod::NeuropodValue> float_tensor_1D =
        allocator->allocate_tensor({3}, neuropod::FLOAT_TENSOR);
    float_tensor_1D->as_typed_tensor<float>()->copy_from({0.0, 0.1, 0.2});

    const std::shared_ptr<neuropod::NeuropodValue> string_tensor_2D =
        allocator->allocate_tensor({2, 2}, neuropod::STRING_TENSOR);
    string_tensor_2D->as_typed_tensor<std::string>()->set({"A", "B", "C", "D"});

    const std::shared_ptr<neuropod::NeuropodValue> int_tensor_2D =
        allocator->allocate_tensor({2, 3}, neuropod::INT32_TENSOR);
    int_tensor_2D->as_typed_tensor<int32_t>()->copy_from({0, 1, 2, 3, 4, 5});

    // Create a map
    neuropod::NeuropodValueMap data;
    data["int"]    = int_tensor_2D;
    data["string"] = string_tensor_2D;
    data["float"]  = float_tensor_1D;

    // Serialize the map
    std::stringstream ss;
    neuropod::serialize(ss, data);

    // Deserialize the map
    ss.seekg(0, std::ios::beg);
    const auto deserialized = neuropod::deserialize<neuropod::NeuropodValueMap>(ss, *allocator);

    EXPECT_EQ(*int_tensor_2D, *deserialized.at("int"));
    EXPECT_EQ(*string_tensor_2D, *deserialized.at("string"));
    EXPECT_EQ(*float_tensor_1D, *deserialized.at("float"));
}
