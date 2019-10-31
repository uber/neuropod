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

size_t compute_num_elements(const std::vector<int64_t> &dims)
{
    // Get the number of elements in the tensor by multiplying all the dims together
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
}

} // namespace

NeuropodTensor::NeuropodTensor(TensorType tensor_type, const std::vector<int64_t> dims)
    : NeuropodValue(true),
      tensor_type_(tensor_type),
      dims_(dims),
      strides_(compute_strides(dims)),
      num_elements_(compute_num_elements(dims))
{
}

// This checks equality of contents, not of addresses or underlying implementations
// (e.g. comparing a TorchNeuropodTensor and a TensorflowNeuropodTensor with identical
// shapes, types, and content would return true)
bool NeuropodValue::operator==(const NeuropodValue &other) const
{
    if (!is_tensor_ || !other.is_tensor_)
    {
        NEUROPOD_ERROR(
            "The equality operator is currently only defined for tensor types. "
            "Tried to compare a value that is not a tensor."
        );
    }

    return (*as_tensor()) == (*other.as_tensor());
}

// This checks equality of contents, not of addresses or underlying implementations
// (e.g. comparing a TorchNeuropodTensor and a TensorflowNeuropodTensor with identical
// shapes, types, and content would return true)
bool NeuropodTensor::operator==(const NeuropodTensor &other) const
{
    // Make sure they have the same type
    if (get_tensor_type() != other.get_tensor_type())
    {
        return false;
    }

    // Make sure they have the same dims
    if (get_dims() != other.get_dims())
    {
        return false;
    }

    // String tensor equality is different than numeric equality
    if (get_tensor_type() == STRING_TENSOR)
    {
        // TODO(vip): optimize
        return as_typed_tensor<std::string>()->get_data_as_vector() == other.as_typed_tensor<std::string>()->get_data_as_vector();
    }

    // Compare the contents of the tensor
    const auto num_bytes = get_num_elements() * get_bytes_per_element();

    const void * first  = get_untyped_data_ptr();
    const void * second = other.get_untyped_data_ptr();

    // Optimization for comparing tensors pointing to the same memory
    if (first == second)
    {
        return true;
    }

    return memcmp(first, second, num_bytes) == 0;
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
