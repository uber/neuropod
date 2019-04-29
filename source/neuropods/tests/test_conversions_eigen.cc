//
// Uber, Inc. (c) 2018
//

#include "gtest/gtest.h"

#include "neuropods/backends/test_backend/test_neuropod_backend.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/conversions/eigen.hh"

#include <gmock/gmock.h>

class int32_tensor_fixture : public ::testing::Test
{
public:
    int32_tensor_fixture()
    {
        untyped_tensor       = test_backend_.get_tensor_allocator()->allocate_tensor({ROWS}, neuropods::INT32_TENSOR);
        const_untyped_tensor = untyped_tensor.get();

        tensor       = untyped_tensor->as_typed_tensor<int32_t>();
        const_tensor = tensor;

        for (size_t i = 0; i < ROWS; ++i)
        {
            (*tensor)[i]       = i;
            expected_vector[i] = i;
        }
    }

protected:
    static constexpr int                           ROWS = 4;
    neuropods::TestNeuropodBackend                 test_backend_;
    std::unique_ptr<neuropods::NeuropodTensor>     untyped_tensor;
    const neuropods::NeuropodTensor *              const_untyped_tensor;
    neuropods::TypedNeuropodTensor<int32_t> *      tensor;
    const neuropods::TypedNeuropodTensor<int32_t> *const_tensor;
    Eigen::Vector4i                                expected_vector;
};

class int32_matrix_fixture : public ::testing::Test
{
public:
    int32_matrix_fixture()
    {
        untyped_tensor       = test_backend_.get_tensor_allocator()->allocate_tensor({ROWS, COLS}, neuropods::INT32_TENSOR);
        const_untyped_tensor = untyped_tensor.get();

        tensor       = untyped_tensor->as_typed_tensor<int32_t>();
        const_tensor = tensor;

        int i = 0;
        for (size_t row = 0; row < untyped_tensor->get_dims()[0]; ++row)
        {
            for (size_t col = 0; col < untyped_tensor->get_dims()[1]; ++col)
            {
                (*tensor)(row, col)       = i;
                expected_matrix(row, col) = i;
                ++i;
            }
        }
    }

protected:
    static constexpr int                           ROWS = 5;
    static constexpr int                           COLS = 3;
    neuropods::TestNeuropodBackend                 test_backend_;
    std::unique_ptr<neuropods::NeuropodTensor>     untyped_tensor;
    const neuropods::NeuropodTensor *              const_untyped_tensor;
    neuropods::TypedNeuropodTensor<int32_t> *      tensor;
    const neuropods::TypedNeuropodTensor<int32_t> *const_tensor;
    Eigen::Matrix<int32_t, ROWS, COLS>             expected_matrix;
};

TEST_F(int32_tensor_fixture, untyped_vector_as_eigen)
{
    auto actual = neuropods::as_eigen<int32_t>(*untyped_tensor);
    EXPECT_EQ(actual, expected_vector);
}

TEST_F(int32_tensor_fixture, typed_vector_as_eigen)
{
    auto actual = neuropods::as_eigen(*tensor);
    EXPECT_EQ(actual, expected_vector);
    // Make sure we are using the same underlying buffer and the neuropod tensor data is editable
    actual(0) = 42;
    EXPECT_EQ((*tensor)[0], 42);
}

TEST_F(int32_tensor_fixture, untyped_vector_as_eigen_type_mismatch)
{
    EXPECT_THROW(neuropods::as_eigen<float>(*untyped_tensor), std::runtime_error);
}

TEST_F(int32_tensor_fixture, const_untyped_vector_as_eigen)
{
    const auto actual = neuropods::as_eigen<int32_t>(*const_untyped_tensor);
    EXPECT_EQ(actual, expected_vector);
}

TEST_F(int32_tensor_fixture, const_typed_vector_as_eigen)
{
    const auto actual = neuropods::as_eigen(*const_tensor);
    EXPECT_EQ(actual, expected_vector);
    (*tensor)[0] = 42;
    EXPECT_EQ(actual(0), 42);
}

TEST_F(int32_tensor_fixture, const_untyped_vector_as_eigen_type_mismatch)
{
    EXPECT_THROW(neuropods::as_eigen<float>(*const_untyped_tensor), std::runtime_error);
}

/////////////////////////

TEST_F(int32_matrix_fixture, untyped_matrix_as_eigen)
{
    auto actual = neuropods::as_eigen<int32_t>(*untyped_tensor);
    EXPECT_EQ(actual, expected_matrix);
}

TEST_F(int32_matrix_fixture, typed_vector_as_eigen)
{
    auto actual = neuropods::as_eigen(*tensor);
    EXPECT_EQ(actual, expected_matrix);
}

TEST_F(int32_matrix_fixture, untyped_vector_as_eigen_type_mismatch)
{
    EXPECT_THROW(neuropods::as_eigen<float>(*untyped_tensor), std::runtime_error);
}

TEST_F(int32_matrix_fixture, const_untyped_vector_as_eigen)
{
    const auto actual = neuropods::as_eigen<int32_t>(*const_untyped_tensor);
    EXPECT_EQ(actual, expected_matrix);
}

TEST_F(int32_matrix_fixture, const_typed_vector_as_eigen)
{
    const auto actual = neuropods::as_eigen(*const_tensor);
    EXPECT_EQ(actual, expected_matrix);
}

TEST_F(int32_matrix_fixture, const_untyped_vector_as_eigen_type_mismatch)
{
    EXPECT_THROW(neuropods::as_eigen<float>(*const_untyped_tensor), std::runtime_error);
}

TEST_F(int32_matrix_fixture, higher_rank)
{
    untyped_tensor = test_backend_.get_tensor_allocator()->allocate_tensor({5, 5, 5}, neuropods::INT32_TENSOR);
    EXPECT_THROW(neuropods::as_eigen<int32_t>(*untyped_tensor), std::runtime_error);
}
