//
// Uber, Inc. (c) 2019
//

#include <iostream>
#include <string>
#include <vector>

#include "neuropods/neuropods.hh"

namespace neuropods
{
namespace detail
{

// Do not use this visitor directly. Use `wrap_existing_tensor` below.
// Given a Neuropod and a tensor, create a new tensor by wrapping data from the input tensor
struct wrap_tensor_data_with_neuropod_visitor : public NeuropodTensorVisitor<std::shared_ptr<NeuropodTensor>>
{
    template <typename T>
    std::shared_ptr<NeuropodTensor> operator()(TypedNeuropodTensor<T> *tensor, Neuropod &neuropod, const Deleter &deleter) const
    {
        // Get a pointer to the data in the original tensor
        T * data = tensor->get_raw_data_ptr();

        // Create a new tensor that wraps the same data
        return neuropod.tensor_from_memory(
            tensor->get_dims(),
            data,
            deleter
        );
    }

    std::shared_ptr<NeuropodTensor> operator()(TypedNeuropodTensor<std::string> *tensor, Neuropod &neuropod, const Deleter &deleter) const
    {
        NEUROPOD_ERROR("It is not currently possible to wrap string tensors.");
    }
};

// Do not use this visitor directly. Use `wrap_existing_tensor` below.
// Given a tensor, create a new tensor of a specific type by wrapping data from the input tensor
template <template <class> class TensorClass>
struct wrap_tensor_data_with_tensor_visitor : public NeuropodTensorVisitor<std::shared_ptr<NeuropodTensor>>
{
    template <typename T>
    std::shared_ptr<NeuropodTensor> operator()(TypedNeuropodTensor<T> *tensor, const Deleter &deleter) const
    {
        T * data = tensor->get_raw_data_ptr();
        return make_tensor_no_string<TensorClass>(
            tensor->get_tensor_type(),
            tensor->get_dims(),
            data,
            deleter
        );
    }

    std::shared_ptr<NeuropodTensor> operator()(TypedNeuropodTensor<std::string> *tensor, const Deleter &deleter) const
    {
        NEUROPOD_ERROR("It is not currently possible to wrap string tensors.");
    }
};

} // namespace

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
    const auto deleter = [tensor](void * unused) {};

    // Create a "native" tensor with the data in the provided tensor
    // This doesn't do a copy; it just wraps the data and passes it to the
    // underlying backend
    return tensor->apply_visitor(detail::wrap_tensor_data_with_neuropod_visitor{}, neuropod, deleter);
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
    const auto deleter = [tensor](void * unused) {};

    // Create a new tensor of the specified type with the data in the provided tensor
    // This doesn't do a copy; it just wraps the data and passes it to the
    // underlying backend
    return tensor->apply_visitor(detail::wrap_tensor_data_with_tensor_visitor<TensorClass>{}, deleter);
}

} // namespace neuropods
