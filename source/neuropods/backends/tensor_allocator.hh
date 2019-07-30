//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/internal/tensor_types.hh"

#include <memory>
#include <string>
#include <vector>

namespace neuropods
{

// An base class used to allocate tensors
class NeuropodTensorAllocator
{
public:
    virtual ~NeuropodTensorAllocator() {}

    // Allocate a tensor of a specific type
    virtual std::unique_ptr<NeuropodTensor> allocate_tensor(const std::vector<int64_t> &input_dims,
                                                            TensorType                  tensor_type) = 0;

    // Allocate a tensor of a specific type and wrap existing memory.
    // Note: Some backends may have specific alignment requirements (e.g. tensorflow).
    // To support all the built-in backends, `data` should be aligned to 64 bytes.
    // `deleter` will be called with a pointer to `data` when the tensor is
    // deallocated
    virtual std::unique_ptr<NeuropodTensor> tensor_from_memory(const std::vector<int64_t> &input_dims,
                                                               TensorType                  tensor_type,
                                                               void *                      data,
                                                               const Deleter &             deleter) = 0;

    // Templated version of `allocate_tensor`
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> allocate_tensor(const std::vector<int64_t> &input_dims)
    {
        std::shared_ptr<NeuropodTensor> tensor = this->allocate_tensor(input_dims, get_tensor_type_from_cpp<T>());

        return std::dynamic_pointer_cast<TypedNeuropodTensor<T>>(tensor);
    }

    // Templated version of `tensor_from_memory`
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> tensor_from_memory(const std::vector<int64_t> &input_dims,
                                                               T *                         data,
                                                               const Deleter &             deleter)
    {
        std::shared_ptr<NeuropodTensor> tensor =
            this->tensor_from_memory(input_dims, get_tensor_type_from_cpp<T>(), data, deleter);

        return std::dynamic_pointer_cast<TypedNeuropodTensor<T>>(tensor);
    }
};

// A default allocator
template <template <class> class TensorImpl>
class DefaultTensorAllocator : public NeuropodTensorAllocator
{
public:
    std::unique_ptr<NeuropodTensor> allocate_tensor(const std::vector<int64_t> &input_dims, TensorType tensor_type)
    {
        return make_tensor<TensorImpl>(tensor_type, input_dims);
    }

    std::unique_ptr<NeuropodTensor> tensor_from_memory(const std::vector<int64_t> &input_dims,
                                                       TensorType                  tensor_type,
                                                       void *                      data,
                                                       const Deleter &             deleter)
    {
        return make_tensor_no_string<TensorImpl>(tensor_type, input_dims, data, deleter);
    }
};

} // namespace neuropods
