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
#include "neuropod/conversions/eigen.hh"
#include "neuropod/core/generic_tensor.hh"
#include "neuropod/internal/neuropod_tensor.hh"

#include <gmock/gmock.h>

class int32_tensor_fixture : public ::testing::Test
{
public:
    int32_tensor_fixture()
    {
        untyped_tensor = neuropod::get_generic_tensor_allocator()->allocate_tensor({ROWS}, neuropod::INT32_TENSOR);
        const_untyped_tensor = untyped_tensor.get();

        tensor       = untyped_tensor->as_typed_tensor<int32_t>();
        const_tensor = tensor;

        auto accessor = tensor->accessor<1>();
        for (size_t i = 0; i < ROWS; ++i)
        {
            accessor[i]        = i;
            expected_vector[i] = i;
        }
    }

protected:
    static constexpr int                          ROWS = 4;
    std::unique_ptr<neuropod::NeuropodTensor>     untyped_tensor;
    const neuropod::NeuropodTensor *              const_untyped_tensor;
    neuropod::TypedNeuropodTensor<int32_t> *      tensor;
    const neuropod::TypedNeuropodTensor<int32_t> *const_tensor;
    Eigen::Vector4i                               expected_vector;
};

class int32_matrix_fixture : public ::testing::Test
{
public:
    int32_matrix_fixture()
    {
        untyped_tensor =
            neuropod::get_generic_tensor_allocator()->allocate_tensor({ROWS, COLS}, neuropod::INT32_TENSOR);
        const_untyped_tensor = untyped_tensor.get();

        tensor       = untyped_tensor->as_typed_tensor<int32_t>();
        const_tensor = tensor;

        auto accessor = tensor->accessor<2>();
        int  i        = 0;
        for (size_t row = 0; row < untyped_tensor->get_dims()[0]; ++row)
        {
            for (size_t col = 0; col < untyped_tensor->get_dims()[1]; ++col)
            {
                accessor[row][col]        = i;
                expected_matrix(row, col) = i;
                ++i;
            }
        }
    }

protected:
    static constexpr int                          ROWS = 5;
    static constexpr int                          COLS = 3;
    std::unique_ptr<neuropod::NeuropodTensor>     untyped_tensor;
    const neuropod::NeuropodTensor *              const_untyped_tensor;
    neuropod::TypedNeuropodTensor<int32_t> *      tensor;
    const neuropod::TypedNeuropodTensor<int32_t> *const_tensor;
    Eigen::Matrix<int32_t, ROWS, COLS>            expected_matrix;
};

TEST_F(int32_tensor_fixture, untyped_vector_as_eigen)
{
    auto actual = neuropod::as_eigen<int32_t>(*untyped_tensor);
    EXPECT_EQ(actual, expected_vector);
}

TEST_F(int32_tensor_fixture, typed_vector_as_eigen)
{
    auto actual = neuropod::as_eigen(*tensor);
    EXPECT_EQ(actual, expected_vector);
    // Make sure we are using the same underlying buffer and the neuropod tensor data is editable
    actual(0) = 42;
    EXPECT_EQ(tensor->accessor<1>()[0], 42);
}

TEST_F(int32_tensor_fixture, untyped_vector_as_eigen_type_mismatch)
{
    EXPECT_THROW(neuropod::as_eigen<float>(*untyped_tensor), std::runtime_error);
}

TEST_F(int32_tensor_fixture, const_untyped_vector_as_eigen)
{
    const auto actual = neuropod::as_eigen<int32_t>(*const_untyped_tensor);
    EXPECT_EQ(actual, expected_vector);
}

TEST_F(int32_tensor_fixture, const_typed_vector_as_eigen)
{
    const auto actual = neuropod::as_eigen(*const_tensor);
    EXPECT_EQ(actual, expected_vector);
    tensor->accessor<1>()[0] = 42;
    EXPECT_EQ(actual(0), 42);
}

TEST_F(int32_tensor_fixture, const_untyped_vector_as_eigen_type_mismatch)
{
    EXPECT_THROW(neuropod::as_eigen<float>(*const_untyped_tensor), std::runtime_error);
}

/////////////////////////

TEST_F(int32_matrix_fixture, untyped_matrix_as_eigen)
{
    auto actual = neuropod::as_eigen<int32_t>(*untyped_tensor);
    EXPECT_EQ(actual, expected_matrix);
}

TEST_F(int32_matrix_fixture, typed_vector_as_eigen)
{
    auto actual = neuropod::as_eigen(*tensor);
    EXPECT_EQ(actual, expected_matrix);
}

TEST_F(int32_matrix_fixture, untyped_vector_as_eigen_type_mismatch)
{
    EXPECT_THROW(neuropod::as_eigen<float>(*untyped_tensor), std::runtime_error);
}

TEST_F(int32_matrix_fixture, const_untyped_vector_as_eigen)
{
    const auto actual = neuropod::as_eigen<int32_t>(*const_untyped_tensor);
    EXPECT_EQ(actual, expected_matrix);
}

TEST_F(int32_matrix_fixture, const_typed_vector_as_eigen)
{
    const auto actual = neuropod::as_eigen(*const_tensor);
    EXPECT_EQ(actual, expected_matrix);
}

TEST_F(int32_matrix_fixture, const_untyped_vector_as_eigen_type_mismatch)
{
    EXPECT_THROW(neuropod::as_eigen<float>(*const_untyped_tensor), std::runtime_error);
}

TEST_F(int32_matrix_fixture, higher_rank)
{
    untyped_tensor = neuropod::get_generic_tensor_allocator()->allocate_tensor({5, 5, 5}, neuropod::INT32_TENSOR);
    EXPECT_THROW(neuropod::as_eigen<int32_t>(*untyped_tensor), std::runtime_error);
}
