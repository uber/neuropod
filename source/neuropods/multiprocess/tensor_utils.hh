//
// Uber, Inc. (c) 2019
//

#include "neuropods/neuropods.hh"
#include "neuropods/internal/neuropod_tensor_raw_data_access.hh"

#include <iostream>
#include <string>
#include <vector>

namespace neuropods
{

// Given a NeuropodTensor, wrap the underlying data with a newly created tensor
// This is useful for serialization and to wrap and/or copy tensors between backends.
// For example, if you had a TorchNeuropodTensor and you wanted to get a tensor compatible
// with `neuropod` without making a copy, you could use this function
std::shared_ptr<NeuropodTensor> wrap_existing_tensor(Neuropod &neuropod, std::shared_ptr<NeuropodTensor> tensor)
{
    // Whenever you're wrapping existing memory, it is very important to make sure that the data
    // being wrapped does not get deleted before the underlying DL framework is done with the
    // created tensor.
    //
    // In this case, we're capturing `tensor` in the deleter below. This ensures that the tensor
    // doesn't get deallocated until we're done with the new tensor.
    const auto  deleter = [tensor](void *unused) {};
    const auto &tensor_type = tensor->get_tensor_type();
    if (tensor_type == STRING_TENSOR)
    {
        NEUROPOD_ERROR("It is not currently possible to wrap string tensors.");
    }
    else
    {
        // Get a pointer to the data in the original tensor
        void *data = NeuropodTensorRawDataAccess::get_untyped_data_ptr(*tensor);

        // Create a "native" tensor with the data in the provided tensor
        // This doesn't do a copy; it just wraps the data and passes it to the
        // underlying backend
        return neuropod.get_tensor_allocator()->tensor_from_memory(tensor->get_dims(), tensor_type, data, deleter);
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
    const auto  deleter = [tensor](void *unused) {};
    const auto &tensor_type = tensor->get_tensor_type();
    if (tensor_type == STRING_TENSOR)
    {
        NEUROPOD_ERROR("It is not currently possible to wrap string tensors.");
    }
    else
    {
        void *data = NeuropodTensorRawDataAccess::get_untyped_data_ptr(*tensor);

        // Create a new tensor of the specified type with the data in the provided tensor
        // This doesn't do a copy; it just wraps the data and passes it to the
        // underlying backend
        return make_tensor_no_string<TensorClass>(tensor_type, tensor->get_dims(), data, deleter);
    }
}

} // namespace neuropods
