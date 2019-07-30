//
// Uber, Inc. (c) 2018
//

#include "gtest/gtest.h"
#include "neuropods/backends/test_backend/test_neuropod_backend.hh"
#include "neuropods/internal/neuropod_tensor.hh"

#include <gmock/gmock.h>

using ::testing::HasSubstr;

class uint8_tensor_fixture : public ::testing::Test
{
public:
    uint8_tensor_fixture()
    {
        untyped_tensor =
            test_backend_.get_tensor_allocator()->allocate_tensor({EXPECTED_SIZE}, neuropods::UINT8_TENSOR);
        const_untyped_tensor = untyped_tensor.get();

        tensor       = untyped_tensor->as_typed_tensor<uint8_t>();
        const_tensor = tensor;

        for (size_t i = 0; i < EXPECTED_SIZE; ++i)
        {
            (*tensor)[i] = i;
        }
    }

protected:
    static int                                     EXPECTED_SIZE;
    neuropods::TestNeuropodBackend                 test_backend_;
    std::unique_ptr<neuropods::NeuropodTensor>     untyped_tensor;
    const neuropods::NeuropodTensor *              const_untyped_tensor;
    neuropods::TypedNeuropodTensor<uint8_t> *      tensor;
    const neuropods::TypedNeuropodTensor<uint8_t> *const_tensor;
};
int uint8_tensor_fixture::EXPECTED_SIZE = 10;

class uint8_scalar_fixture : public ::testing::Test
{
public:
    uint8_scalar_fixture()
    {
        untyped_tensor = test_backend_.get_tensor_allocator()->allocate_tensor({1}, neuropods::UINT8_TENSOR);
        untyped_tensor->as_scalar<uint8_t>() = 42;
        const_untyped_tensor                 = untyped_tensor.get();
        tensor                               = untyped_tensor->as_typed_tensor<uint8_t>();
        const_tensor                         = tensor;
    }

protected:
    neuropods::TestNeuropodBackend                 test_backend_;
    std::unique_ptr<neuropods::NeuropodTensor>     untyped_tensor;
    const neuropods::NeuropodTensor *              const_untyped_tensor;
    neuropods::TypedNeuropodTensor<uint8_t> *      tensor;
    const neuropods::TypedNeuropodTensor<uint8_t> *const_tensor;
};

TEST(test_stream_operator, untyped_tensor)
{
    std::stringstream              ss;
    neuropods::TestNeuropodBackend test_backend;
    const auto untyped_tensor = test_backend.get_tensor_allocator()->allocate_tensor({3}, neuropods::UINT8_TENSOR);
    ss << *untyped_tensor;

    EXPECT_THAT(ss.str(), HasSubstr("NeuropodTensor"));
}

TEST(test_stream_operator, typed_tensor)
{
    std::stringstream              ss;
    neuropods::TestNeuropodBackend test_backend;
    auto untyped_tensor = test_backend.get_tensor_allocator()->allocate_tensor({3}, neuropods::UINT8_TENSOR);

    auto &typed_tensor = *untyped_tensor->as_typed_tensor<uint8_t>();

    typed_tensor[0] = 10;
    typed_tensor[1] = 11;
    typed_tensor[2] = 12;

    ss << typed_tensor;
    EXPECT_THAT(ss.str(), HasSubstr("NeuropodTensor"));
    EXPECT_THAT(ss.str(), HasSubstr("[10, 11, 12]"));
}

TEST(test_stream_operator, typed_float_tensor)
{
    std::stringstream              ss;
    neuropods::TestNeuropodBackend test_backend;
    constexpr int                  TENSOR_SIZE = 8;
    auto untyped_tensor = test_backend.get_tensor_allocator()->allocate_tensor({TENSOR_SIZE}, neuropods::FLOAT_TENSOR);

    auto &typed_tensor = *untyped_tensor->as_typed_tensor<float>();

    for (int i = 0; i < TENSOR_SIZE; ++i)
    {
        typed_tensor[i] = i + 0.5;
    }

    ss << typed_tensor;
    EXPECT_THAT(ss.str(), HasSubstr("NeuropodTensor"));
    EXPECT_THAT(ss.str(), HasSubstr("[0.5, 1.5, 2.5 ... 5.5, 6.5, 7.5]"));
}

TEST(test_typed_neuropod_tensor, downcast_failulre)
{
    neuropods::TestNeuropodBackend test_backend;
    constexpr int                  TENSOR_SIZE = 8;
    auto untyped_tensor = test_backend.get_tensor_allocator()->allocate_tensor({TENSOR_SIZE}, neuropods::FLOAT_TENSOR);

    EXPECT_THROW(untyped_tensor->as_typed_tensor<int8_t>(), std::runtime_error);
}

TEST_F(uint8_tensor_fixture, forloop)
{
    uint8_t i = 0;
    for (auto element : *tensor)
    {
        EXPECT_EQ(element, i);
        ++i;
    }
    EXPECT_EQ(EXPECTED_SIZE, i);
}

TEST_F(uint8_tensor_fixture, forloop_const_tensor)
{
    uint8_t i = 0;
    for (const auto &element : *const_tensor)
    {
        EXPECT_EQ(element, i);
        ++i;
    }
    EXPECT_EQ(EXPECTED_SIZE, i);
}

TEST_F(uint8_tensor_fixture, forloop_const_reference)
{
    uint8_t i = 0;
    for (const auto &element : *tensor)
    {
        EXPECT_EQ(element, i);
        ++i;
    }
    EXPECT_EQ(EXPECTED_SIZE, i);
}

TEST_F(uint8_tensor_fixture, can_not_cast_to_scalar)
{
    EXPECT_THROW(untyped_tensor->as_scalar<uint16_t>(), std::runtime_error);
}

TEST_F(uint8_tensor_fixture, wrong_dimensions)
{
    EXPECT_THROW(untyped_tensor->as_scalar<uint8_t>(), std::runtime_error);
}

TEST_F(uint8_tensor_fixture, const_wrong_dimensions)
{
    EXPECT_THROW(const_untyped_tensor->as_scalar<uint8_t>(), std::runtime_error);
}

TEST_F(uint8_scalar_fixture, non_const)
{
    untyped_tensor->as_scalar<uint8_t>() = 10;
    EXPECT_EQ(untyped_tensor->as_scalar<uint8_t>(), 10);
}

TEST_F(uint8_scalar_fixture, const_access)
{
    const auto &actual = const_untyped_tensor->as_scalar<uint8_t>();
    EXPECT_EQ(actual, 42);
}

TEST_F(uint8_scalar_fixture, wrong_type)
{
    EXPECT_THROW(const_untyped_tensor->as_scalar<uint16_t>(), std::runtime_error);
}

TEST_F(uint8_scalar_fixture, typed_non_const)
{
    tensor->as_scalar() = 10;
    EXPECT_EQ(tensor->as_scalar(), 10);
}

TEST_F(uint8_scalar_fixture, typed_const_access)
{
    const auto &actual = const_tensor->as_scalar();
    EXPECT_EQ(actual, 42);
}
