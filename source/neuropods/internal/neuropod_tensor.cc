//
// Uber, Inc. (c) 2019
//

#include "neuropods/internal/neuropod_tensor.hh"

namespace neuropods
{

namespace
{

std::vector<int64_t> compute_strides(const std::vector<int64_t> &dims)
{
    // To compute strides from the dimensions, we want to do the following:
    // For an index i, if i is the last element:
    //     strides[i] = 1
    // else
    //     strides[i] = product(dims[i + 1:])
    std::vector<int64_t> out(dims.size());

    int64_t running_product = 1;
    for (int i = out.size() - 1; i >= 0; i--)
    {
        // Set the stride
        out[i] = running_product;

        // Update the running product
        running_product *= dims[i];
    }

    return out;
}

} // namespace

NeuropodTensor::NeuropodTensor(TensorType tensor_type, const std::vector<int64_t> dims)
    : NeuropodValue(true), tensor_type_(tensor_type), dims_(dims), strides_(compute_strides(dims))
{
}

NeuropodTensor *NeuropodValue::as_tensor()
{
    assure_tensor();
    return dynamic_cast<NeuropodTensor *>(this);
}

const NeuropodTensor *NeuropodValue::as_tensor() const
{
    assure_tensor();
    return dynamic_cast<const NeuropodTensor *>(this);
}

template <typename T>
TypedNeuropodTensor<T> *NeuropodValue::as_typed_tensor()
{
    return this->as_tensor()->as_typed_tensor<T>();
}

template <typename T>
const TypedNeuropodTensor<T> *NeuropodValue::as_typed_tensor() const
{
    return this->as_tensor()->as_typed_tensor<T>();
}

#define INIT_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE)                            \
    template TypedNeuropodTensor<CPP_TYPE> *      NeuropodValue::as_typed_tensor(); \
    template const TypedNeuropodTensor<CPP_TYPE> *NeuropodValue::as_typed_tensor() const;

FOR_EACH_TYPE_MAPPING_INCLUDING_STRING(INIT_TEMPLATES_FOR_TYPE);

} // namespace neuropods
