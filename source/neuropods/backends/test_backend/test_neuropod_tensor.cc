//
// Uber, Inc. (c) 2018
//

#include "test_neuropod_tensor.hh"

namespace neuropods
{

template <typename T>
TestNeuropodTensor<T>::TestNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
    : TypedNeuropodTensor<T>(name, dims)
{
    data_ = malloc(this->get_num_elements() * sizeof(T));
}

template <typename T>
TestNeuropodTensor<T>::~TestNeuropodTensor()
{
    free(data_);
}

template <typename T>
T *TestNeuropodTensor<T>::get_raw_data_ptr()
{
    return static_cast<T *>(data_);
}

// Instantiate the templates
#define INIT_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE) template class TestNeuropodTensor<CPP_TYPE>;

FOR_EACH_TYPE_MAPPING(INIT_TEMPLATES_FOR_TYPE);
} // namespace neuropods