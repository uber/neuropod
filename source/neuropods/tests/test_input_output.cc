//
// Uber, Inc. (c) 2018
//

#include "gtest/gtest.h"

#include "neuropods/backends/test_backend/test_neuropod_backend.hh"
#include "neuropods/internal/tensor_store.hh"
#include "neuropods/neuropod_input_builder.hh"
#include "neuropods/neuropod_output_data.hh"

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
    // Get a pointer to the internal data
    const auto tensor_data_ptr = boost::get<T *>(tensor->get_data_ptr());

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

// Creates tensors using various input methods
// and validates the output of the builder
TEST(test_input_builder, add_tensors_and_build)
{
    neuropods::NeuropodInputBuilder builder(std::make_shared<neuropods::TestNeuropodBackend>());

    builder.add_tensor("a", a_data, a_shape).add_tensor("b", b_data, b_shape).add_tensor("c", c_data, 4, c_shape);

    // Test using allocate
    {
        double *data = builder.allocate_tensor<double>("d", 4, d_shape);
        memcpy(data, d_data, 4 * sizeof(double));
    }


    auto tensor_store = builder.build();

    // Validate the internal state for a
    {
        const auto ten = tensor_store->find("a");
        check_tensor_eq_ptr(ten, a_data.data(), a_data.size());
        check_vectors_eq(ten->get_dims(), a_shape);
    }

    // Validate the internal state for b
    {
        const auto ten = tensor_store->find("b");
        check_tensor_eq_ptr(ten, b_data.data(), b_data.size());
        check_vectors_eq(ten->get_dims(), b_shape);
    }

    // Validate the internal state for c
    {
        const auto ten = tensor_store->find("c");
        check_tensor_eq_ptr(ten, c_data, 4);
        check_vectors_eq(ten->get_dims(), c_shape);
    }

    // Validate the internal state for d
    {
        const auto ten = tensor_store->find("d");
        check_tensor_eq_ptr(ten, d_data, 4);
        check_vectors_eq(ten->get_dims(), d_shape);
    }
}


// Verifies that NeuropodOutputData is correctly
// reading the tensors in a TensorStore
TEST(test_output_data, verify_tensors)
{
    neuropods::NeuropodInputBuilder builder(std::make_shared<neuropods::TestNeuropodBackend>());

    auto tensor_store = builder.add_tensor("a", a_data, a_shape)
                            .add_tensor("b", b_data, b_shape)
                            .add_tensor("c", c_data, 4, c_shape)
                            .add_tensor("d", d_data, 4, d_shape)
                            .build();

    // Create a NeuropodOutputData from the tensor store
    neuropods::NeuropodOutputData output_data(std::move(tensor_store));

    // Validate the internal state for a
    {
        // Pointer and size
        const int32_t *pointer;
        size_t         size;
        output_data.get_data_pointer_and_size("a", pointer, size);
        check_ptrs_eq(pointer, size, a_data.data(), a_data.size());

        // Vector
        std::vector<int32_t> vec = output_data.get_data_as_vector<int32_t>("a");
        check_vectors_eq(vec, a_data);

        // Shape
        check_vectors_eq(output_data.get_shape("a"), a_shape);
    }

    // Validate the internal state for b
    {
        // Pointer and size
        const int64_t *pointer;
        size_t         size;
        output_data.get_data_pointer_and_size("b", pointer, size);
        check_ptrs_eq(pointer, size, b_data.data(), b_data.size());

        // Vector
        std::vector<int64_t> vec = output_data.get_data_as_vector<int64_t>("b");
        check_vectors_eq(vec, b_data);

        // Shape
        check_vectors_eq(output_data.get_shape("b"), b_shape);
    }

    // Validate the internal state for c
    {
        // Pointer and size
        const float *pointer;
        size_t       size;
        output_data.get_data_pointer_and_size("c", pointer, size);
        check_ptrs_eq(pointer, size, c_data, 4);

        // Shape
        check_vectors_eq(output_data.get_shape("c"), c_shape);
    }

    // Validate the internal state for d
    {
        // Pointer and size
        const double *pointer;
        size_t        size;
        output_data.get_data_pointer_and_size("d", pointer, size);
        check_ptrs_eq(pointer, size, d_data, 4);

        // Shape
        check_vectors_eq(output_data.get_shape("d"), d_shape);
    }
}