//
// Uber, In (c) 2018
//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "neuropods/backends/test_backend/test_neuropod_backend.hh"
#include "neuropods/serialization/numpy.hh"

using ::testing::ElementsAreArray;
using ::testing::ElementsAre;

// template <typename T>
// static void expect_eq(const neuropods::NeuropodTensor& expected, const neuropods::NeuropodTensor& actual)
// {
//     ASSERT_EQ(expected.get_tensor_type(), actual.get_tensor_type());
//     EXPECT_THAT(expected.get_dims(), ElementsAreArray(actual.get_dims()));
//     EXPECT_THAT(expected.as_typed_tensor<T>()->get_data_as_vector(), 
//                 ElementsAreArray(actual.as_typed_tensor<T>()->get_data_as_vector()));
// }

// std::shared_ptr<neuropods::NeuropodTensor> serialize_deserialize(neuropods::NeuropodTensorAllocator &allocator, const neuropods::NeuropodTensor &tensor)
// {
//     // Serialize the tensor
//     std::stringstream ss;

//     neuropods::serialize_tensor(ss, "some_name", tensor);

//     ss.seekg(0, std::ios::beg);

//     // Deserialize the tensor
//     std::string tensor_name;
//     std::shared_ptr<neuropods::NeuropodTensor> out;
//     neuropods::deserialize_tensor(ss, allocator, tensor_name, out);

//     EXPECT_EQ(tensor_name, "some_name");
//     return out;
// }

TEST(test_numpy_serialization, serialize_int_tensor)
{
    neuropods::TestNeuropodBackend backend;

    auto allocator = backend.get_tensor_allocator();

    const auto float_tensor_1D = allocator->allocate_tensor({3}, neuropods::FLOAT_TENSOR);
    float_tensor_1D->as_typed_tensor<float>()->copy_from({0.0, 0.1, 0.2});
    
    neuropods::save_to_npy("/tmp/a.npy", *float_tensor_1D);
    
}

