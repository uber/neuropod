//
// Uber, In (c) 2018
//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "neuropods/backends/test_backend/test_neuropod_backend.hh"
#include "neuropods/serialization/serialization.hh"

using ::testing::ElementsAreArray;
using ::testing::ElementsAre;

template <typename T>
static void expect_eq(const neuropods::NeuropodTensor& expected, const neuropods::NeuropodTensor& actual)
{
    ASSERT_EQ(actual.get_tensor_type(), expected.get_tensor_type());
    EXPECT_THAT(actual.get_dims(), ElementsAreArray(expected.get_dims()));
    EXPECT_THAT(actual.as_typed_tensor<T>()->get_data_as_vector(),
                ElementsAreArray(expected.as_typed_tensor<T>()->get_data_as_vector()));
}

template <typename T>
static void expect_eq(const neuropods::NeuropodTensor& expected, const neuropods::NeuropodValue& actual)
{
    expect_eq<T>(expected, dynamic_cast<const neuropods::NeuropodTensor &>(actual));
}

std::shared_ptr<neuropods::NeuropodTensor> serialize_deserialize(neuropods::NeuropodTensorAllocator &allocator, const neuropods::NeuropodTensor &tensor)
{
    // Serialize the tensor
    std::stringstream ss;
    neuropods::serialize(ss, tensor);

    // Deserialize the tensor
    auto value = neuropods::deserialize(ss, allocator);
    return std::dynamic_pointer_cast<neuropods::NeuropodTensor>(value);
}

TEST(test_allocate_tensor, serialize_string_tensor)
{
    neuropods::TestNeuropodBackend backend;

    std::vector<std::string> expected_data{"A", "B", "C", "D"};

    auto allocator = backend.get_tensor_allocator();

    // Allocate tensors
    const auto tensor_1D = allocator->allocate_tensor({4}, neuropods::STRING_TENSOR);
    tensor_1D->as_typed_tensor<std::string>()->set(expected_data);

    const auto actual_1D = serialize_deserialize(*allocator, *tensor_1D);
    expect_eq<std::string>(*tensor_1D, *actual_1D);

    const auto tensor_2D = allocator->allocate_tensor({2, 2}, neuropods::STRING_TENSOR);
    tensor_2D->as_typed_tensor<std::string>()->set(expected_data);

    const auto actual_2D = serialize_deserialize(*allocator, *tensor_2D);
    expect_eq<std::string>(*tensor_2D, *actual_2D);
}

TEST(test_allocate_tensor, serialize_scalar_tensor)
{
    neuropods::TestNeuropodBackend backend;

    auto allocator = backend.get_tensor_allocator();

    const auto tensor_1D = allocator->allocate_tensor({3}, neuropods::FLOAT_TENSOR);
    tensor_1D->as_typed_tensor<float>()->copy_from({0.0, 0.1, 0.2});

    const auto actual_1D = serialize_deserialize(*allocator, *tensor_1D);
    expect_eq<float>(*tensor_1D, *actual_1D);

    const auto tensor_2D = allocator->allocate_tensor({2, 3}, neuropods::INT32_TENSOR);
    tensor_2D->as_typed_tensor<int32_t>()->copy_from({0, 1, 2, 3, 4, 5});

    const auto actual_2D = serialize_deserialize(*allocator, *tensor_2D);
    expect_eq<int32_t>(*tensor_2D, *actual_2D);

    const auto tensor_3D = allocator->allocate_tensor({2, 2, 2}, neuropods::INT32_TENSOR);
    tensor_3D->as_typed_tensor<int32_t>()->copy_from({0, 1, 2, 3, 4, 5, 6, 7});

    const auto actual_3D = serialize_deserialize(*allocator, *tensor_3D);
    expect_eq<int32_t>(*tensor_3D, *actual_3D);
}

TEST(test_allocate_tensor, multiple_tensors_in_a_stream)
{
    neuropods::TestNeuropodBackend backend;

    auto allocator = backend.get_tensor_allocator();

    const auto float_tensor_1D = allocator->allocate_tensor({3}, neuropods::FLOAT_TENSOR);
    float_tensor_1D->as_typed_tensor<float>()->copy_from({0.0, 0.1, 0.2});

    const auto string_tensor_2D = allocator->allocate_tensor({2, 2}, neuropods::STRING_TENSOR);
    string_tensor_2D->as_typed_tensor<std::string>()->set({"A", "B", "C", "D"});

    const auto int_tensor_2D = allocator->allocate_tensor({2, 3}, neuropods::INT32_TENSOR);
    int_tensor_2D->as_typed_tensor<int32_t>()->copy_from({0, 1, 2, 3, 4, 5});

    std::stringstream ss;

    neuropods::serialize(ss, *int_tensor_2D);
    neuropods::serialize(ss, *string_tensor_2D);
    neuropods::serialize(ss, *float_tensor_1D);

    ss.seekg(0, std::ios::beg);

    // Deserialize the tensor
    expect_eq<int32_t>(*int_tensor_2D, *neuropods::deserialize(ss, *allocator));
    expect_eq<std::string>(*string_tensor_2D, *neuropods::deserialize(ss, *allocator));
    expect_eq<float>(*float_tensor_1D, *neuropods::deserialize(ss, *allocator));
}
