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

#include "neuropod/internal/neuropod_tensor_raw_data_access.hh"
#include "neuropod/neuropod.hh"

#include <iostream>
#include <string>
#include <vector>

namespace neuropod
{

// Given a NeuropodTensor, wrap the underlying data with a newly created tensor
// This is useful for serialization and to wrap and/or copy tensors between backends.
// For example, if you had a TorchNeuropodTensor and you wanted to get a tensor compatible
// with `allocator` without making a copy, you could use this function
std::shared_ptr<NeuropodTensor> wrap_existing_tensor(NeuropodTensorAllocator &       allocator,
                                                     std::shared_ptr<NeuropodTensor> tensor)
{
    // Whenever you're wrapping existing memory, it is very important to make sure that the data
    // being wrapped does not get deleted before the underlying DL framework is done with the
    // created tensor.
    //
    // In this case, we're capturing `tensor` in the deleter below. This ensures that the tensor
    // doesn't get deallocated until we're done with the new tensor.
    const auto  deleter     = [tensor](void *unused) {};
    const auto &tensor_type = tensor->get_tensor_type();
    if (tensor_type == STRING_TENSOR)
    {
        auto out = allocator.allocate_tensor<std::string>(tensor->get_dims());

        // We need to make a copy because it's not possible to generically wrap string tensors
        // (each backend has its own in-memory representation)
        // TODO(vip): optimize
        out->copy_from(tensor->as_typed_tensor<std::string>()->get_data_as_vector());

        return out;
    }
    else
    {
        // Get a pointer to the data in the original tensor
        void *data = internal::NeuropodTensorRawDataAccess::get_untyped_data_ptr(*tensor);

        // Create a "native" tensor with the data in the provided tensor
        // This doesn't do a copy; it just wraps the data and passes it to the
        // underlying backend
        return allocator.tensor_from_memory(tensor->get_dims(), tensor_type, data, deleter);
    }
}

// Given a NeuropodTensor, wrap the underlying data with a newly created tensor
// This is useful for serialization and to wrap and/or copy tensors between backends.
// For example, if you had a TorchNeuropodTensor and you wanted to get a TensorflowNeuropodTensor
// without making a copy, you could use this function.
template <template <class> class TensorClass>
std::shared_ptr<NeuropodTensor> wrap_existing_tensor(std::shared_ptr<NeuropodTensor> tensor)
{
    // Whenever you're wrapping existing memory, it is very important to make sure that the data
    // being wrapped does not get deleted before the underlying DL framework is done with the
    // created tensor.
    //
    // In this case, we're capturing `tensor` in the deleter below. This ensures that the tensor
    // doesn't get deallocated until we're done with the new tensor.
    const auto  deleter     = [tensor](void *unused) {};
    const auto &tensor_type = tensor->get_tensor_type();
    if (tensor_type == STRING_TENSOR)
    {
        auto out = make_tensor<TensorClass>(STRING_TENSOR, tensor->get_dims());

        // We need to make a copy because it's not possible to generically wrap string tensors
        // (each backend has its own in-memory representation)
        // TODO(vip): optimize
        out->template as_typed_tensor<std::string>()->copy_from(
            tensor->as_typed_tensor<std::string>()->get_data_as_vector());

        return out;
    }
    else
    {
        void *data = internal::NeuropodTensorRawDataAccess::get_untyped_data_ptr(*tensor);

        // Create a new tensor of the specified type with the data in the provided tensor
        // This doesn't do a copy; it just wraps the data and passes it to the
        // underlying backend
        return make_tensor_no_string<TensorClass>(tensor_type, tensor->get_dims(), data, deleter);
    }
}

} // namespace neuropod
