//
// Uber, Inc. (c) 2018
//

#include "gtest/gtest.h"
#include "neuropods/backends/test_backend/test_neuropod_backend.hh"
#include "neuropods/serialization/serialization.hh"

namespace
{
// Some test data of various types
const std::vector<int32_t> a_data = {1, 2, 3, 4};
const std::vector<int64_t> b_data = {5, 6, 7, 8};

const float  c_data[] = {9, 10, 11, 12};
const double d_data[] = {13, 14, 15, 16};

const std::vector<int64_t> a_shape = {1, 4};
const std::vector<int64_t> b_shape = {2, 2, 1};
const std::vector<int64_t> c_shape = {4, 1};
const std::vector<int64_t> d_shape = {4};
} // namespace

template <typename T>
void check_ptrs_eq(const T *data, size_t size, const T *target_data, size_t target_size)
{
    // Check that the size and data match expected values
    EXPECT_EQ(size, target_size);
    EXPECT_EQ(memcmp(data, target_data, size * sizeof(T)), 0);
}

template <typename T>
void check_tensor_eq_ptr(const std::shared_ptr<neuropods::NeuropodTensor> &tensor, const T *target_data, size_t size)
{
    // Downcast to a TypedNeuropodTensor so we can get the data pointer
    const auto typed_tensor = tensor->as_typed_tensor<T>();

    // Get a pointer to the internal data
    const auto tensor_data_ptr = typed_tensor->get_raw_data_ptr();

    // Check that the size and data match expected values
    EXPECT_EQ(tensor->get_num_elements(), size);
    EXPECT_EQ(memcmp(tensor_data_ptr, target_data, size * sizeof(T)), 0);
}

template <typename T>
void check_vectors_eq(const std::vector<T> &a, const std::vector<T> &b)
{
    EXPECT_EQ(a.size(), b.size());

    EXPECT_EQ(memcmp(&a[0], &b[0], a.size() * sizeof(T)), 0);
}

std::shared_ptr<neuropods::NeuropodTensor> serialize_deserialize(
    neuropods::NeuropodTensorAllocator &allocator, const std::shared_ptr<neuropods::NeuropodTensor> &tensor)
{
    // Serialize the tensor
    std::stringstream ss;
    neuropods::serialize(ss, *tensor);

    // Deserialize the tensor
    auto value = neuropods::deserialize(ss, allocator);
    return std::dynamic_pointer_cast<neuropods::NeuropodTensor>(value);
}

// Creates tensors using various input methods
// and validates the output
TEST(test_allocate_tensor, add_tensors_and_validate)
{
    neuropods::TestNeuropodBackend backend;
    auto                           allocator = backend.get_tensor_allocator();

    // Allocate tensors
    std::shared_ptr<neuropods::NeuropodTensor> a_ten = allocator->allocate_tensor(a_shape, neuropods::INT32_TENSOR);
    a_ten->as_typed_tensor<int32_t>()->copy_from(a_data);

    std::shared_ptr<neuropods::NeuropodTensor> b_ten = allocator->allocate_tensor(b_shape, neuropods::INT64_TENSOR);
    b_ten->as_typed_tensor<int64_t>()->copy_from(b_data);

    std::shared_ptr<neuropods::NeuropodTensor> c_ten = allocator->allocate_tensor(c_shape, neuropods::FLOAT_TENSOR);
    c_ten->as_typed_tensor<float>()->copy_from(c_data, 4);

    // Wrap existing data
    // TODO(vip): Refactor this test. It's bad practice to have an empty deleter
    // The created tensor should be responsible for deallocating the memory
    std::shared_ptr<neuropods::NeuropodTensor> d_ten = allocator->tensor_from_memory(
        d_shape, neuropods::DOUBLE_TENSOR, const_cast<double *>(d_data), [](void *unused) {});

    // Validate the internal state for a
    {
        const auto ten = serialize_deserialize(*allocator, a_ten);
        check_tensor_eq_ptr(ten, a_data.data(), a_data.size());
        check_vectors_eq(ten->get_dims(), a_shape);
    }

    // Validate the internal state for b
    {
        const auto ten = serialize_deserialize(*allocator, b_ten);
        check_tensor_eq_ptr(ten, b_data.data(), b_data.size());
        check_vectors_eq(ten->get_dims(), b_shape);
    }

    // Validate the internal state for c
    {
        const auto ten = serialize_deserialize(*allocator, c_ten);
        check_tensor_eq_ptr(ten, c_data, 4);
        check_vectors_eq(ten->get_dims(), c_shape);
    }

    // Validate the internal state for d
    {
        const auto ten = serialize_deserialize(*allocator, d_ten);
        check_tensor_eq_ptr(ten, d_data, 4);
        check_vectors_eq(ten->get_dims(), d_shape);
    }
}

// TODO(vip): reenable once the same validation is added to `Neuropod`
// TEST(test_input_builder, adding_tensor_with_the_same_name_should_fail)
// {
//     neuropods::NeuropodInputBuilder builder(std::make_shared<neuropods::TestNeuropodBackend>());
//
//     builder.allocate_tensor<int8_t>({10});
//     EXPECT_THROW(builder.allocate_tensor<int8_t>({10}), std::runtime_error);
// }
