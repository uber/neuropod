//
// Uber, Inc. (c) 2018
//

#include "test_neuropod_tensor.hh"

namespace neuropods
{

namespace
{
size_t get_size_bytes(TensorType tensor_type)
{
#define GET_SIZE(CPP_TYPE, NEUROPOD_TYPE) \
    case NEUROPOD_TYPE:                   \
    {                                     \
        return sizeof(CPP_TYPE);          \
    }

    switch (tensor_type)
    {
        FOR_EACH_TYPE_MAPPING(GET_SIZE)
    }
}
} // namespace

TestNeuropodTensor::TestNeuropodTensor(const std::string &         name,
                                       const std::vector<int64_t> &dims,
                                       TensorType                  tensor_type)
    : NeuropodTensor(name, tensor_type, dims)
{
    data_ = malloc(get_num_elements() * get_size_bytes(tensor_type));
}

TestNeuropodTensor::~TestNeuropodTensor()
{
    free(data_);
}


TensorDataPointer TestNeuropodTensor::get_data_ptr()
{
#define CAST_TENSOR(CPP_TYPE, NEUROPOD_TYPE)   \
    case NEUROPOD_TYPE:                        \
    {                                          \
        return static_cast<CPP_TYPE *>(data_); \
    }

    switch (get_tensor_type())
    {
        FOR_EACH_TYPE_MAPPING(CAST_TENSOR)
    }
}

} // namespace neuropods