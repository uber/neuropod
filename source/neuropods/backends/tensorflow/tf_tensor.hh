//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>

#include <tensorflow/c/c_api.h>

#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/backends/tensorflow/type_utils.hh"

namespace neuropods
{

namespace
{

// Get the shape of a TF tensor
std::vector<int64_t> get_shape(TF_Tensor *tensor)
{
    const int            dims = TF_NumDims(tensor);
    std::vector<int64_t> out;

    for (int curr_dim = 0; curr_dim < dims; curr_dim++)
    {
        out.push_back(TF_Dim(tensor, curr_dim));
    }

    return out;
}

} // namespace


// This class is internal to neuropods and should not be exposed
// to users
template<typename T>
class TensorflowNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<TF_Tensor *>
{
public:
    // Allocate a TF tensor
    TensorflowNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<T>(name, dims),
          tensor(TF_AllocateTensor(get_tf_type_from_neuropod_type(this->get_tensor_type()),
                                   dims.data(),
                                   dims.size(),
                                   this->get_num_elements() * sizeof(T)))
    {
    }

    // Wrap an existing TF tensor
    TensorflowNeuropodTensor(const std::string &name, TF_Tensor *tensor)
        : TypedNeuropodTensor<T>(name, get_shape(tensor)),
          tensor(tensor)
    {
    }

    ~TensorflowNeuropodTensor()
    {
        if (tensor != nullptr)
        {
            TF_DeleteTensor(tensor);
        }
    }

    // Get a pointer to the underlying data
    T *get_raw_data_ptr() { return static_cast<T *>(TF_TensorData(tensor)); }

    TF_Tensor *get_native_data() { return tensor; }

    // The underlying TF tensor
    TF_Tensor *tensor;
};

} // namespace neuropods
