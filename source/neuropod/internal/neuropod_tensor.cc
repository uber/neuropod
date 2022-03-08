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

#include "neuropod/internal/neuropod_tensor.hh"

#include "neuropod/internal/error_utils.hh"

#include <algorithm>

namespace neuropod
{

namespace
{

size_t compute_num_elements(const std::vector<int64_t> &dims)
{
    // Get the number of elements in the tensor by multiplying all the dims together
    return std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<>());
}

} // namespace

namespace detail
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
    for (size_t i = out.size(); i > 0; i--)
    {
        // Set the stride
        out[i - 1] = running_product;

        // Update the running product
        running_product *= dims[i - 1];
    }

    return out;
}

void throw_error_hh(const char *file, int line, const char *function, const std::string &message, TensorType type)
{
    throw_error(file, line, function, message, type);
}

void throw_error_hh(
    const char *file, int line, const char *function, const std::string &message, TensorType type1, TensorType type2)
{
    throw_error(file, line, function, message, type1, type2);
}

} // namespace detail

NeuropodTensor::NeuropodTensor(TensorType tensor_type, const std::vector<int64_t> dims, NeuropodDevice device)
    : NeuropodValue(true),
      tensor_type_(tensor_type),
      dims_(dims),
      strides_(detail::compute_strides(dims)),
      num_elements_(compute_num_elements(dims)),
      device_(device)
{
}

// This checks equality of contents, not of addresses or underlying implementations
// (e.g. comparing a TorchNeuropodTensor and a TensorflowNeuropodTensor with identical
// shapes, types, and content would return true)
bool NeuropodValue::operator==(const NeuropodValue &other) const
{
    if (!is_tensor_ || !other.is_tensor_)
    {
        NEUROPOD_ERROR("The equality operator is currently only defined for tensor types. "
                       "Tried to compare a value that is not a tensor.");
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
        const auto numel = get_num_elements();
        const auto t1    = as_typed_tensor<std::string>();
        const auto t2    = other.as_typed_tensor<std::string>();
        for (size_t i = 0; i < numel; i++)
        {
            if ((*t1)[i] != (*t2)[i])
            {
                return false;
            }
        }

        return true;
    }

    // Compare the contents of the tensor
    const auto num_bytes = get_num_elements() * get_bytes_per_element();

    const void *first  = get_untyped_data_ptr();
    const void *second = other.get_untyped_data_ptr();

    // Optimization for comparing tensors pointing to the same memory
    if (first == second)
    {
        return true;
    }

    return memcmp(first, second, num_bytes) == 0;
}

void NeuropodTensor::assure_device_cpu() const
{
    if (device_ != Device::CPU)
    {
        NEUROPOD_ERROR("Tried to perform an operation on a tensor that expected the tensor to be on CPU. Tensor: {}",
                       *this);
    }
}

void NeuropodTensor::assure_view_compatible_shape(const std::vector<int64_t> &requested_dims) const
{
    // Make sure all provided dimensions are positive
    if (!std::all_of(requested_dims.begin(), requested_dims.end(), [](int64_t i) { return i > 0; }))
    {
        NEUROPOD_ERROR("All dimensions provided to `view` must be positive");
    }

    // Compute the total number of elements
    auto new_num_elements = compute_num_elements(requested_dims);
    auto num_elements     = get_num_elements();

    if (new_num_elements != num_elements)
    {
        NEUROPOD_ERROR("The requested view dimensions are not compatible with the tensor. Requested num "
                       "elements: {}. Actual num elements: {}",
                       new_num_elements,
                       num_elements);
    }
}

NeuropodTensor *NeuropodValue::as_tensor()
{
    assure_tensor();
    auto ten = dynamic_cast<NeuropodTensor *>(this);
    if (ten == nullptr)
    {
        NEUROPOD_ERROR("Casting NeuropodValue to tensor failed");
    }

    return ten;
}

const NeuropodTensor *NeuropodValue::as_tensor() const
{
    assure_tensor();
    auto ten = dynamic_cast<const NeuropodTensor *>(this);
    if (ten == nullptr)
    {
        NEUROPOD_ERROR("Casting NeuropodValue to tensor failed");
    }

    return ten;
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

} // namespace neuropod
