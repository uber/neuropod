//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"

#include <caffe2/core/macros.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>

// If we're not building with a nightly relase of torch,
// set the date to match the date of the official release
#ifndef CAFFE2_NIGHTLY_VERSION
#if CAFFE2_VERSION == 10200
// The date of the official torch 1.2.0 release
#define CAFFE2_NIGHTLY_VERSION 20190808
#endif

#if CAFFE2_VERSION == 10300
// The date of the official torch 1.3.0 release
#define CAFFE2_NIGHTLY_VERSION 20191010
#endif
#endif

namespace neuropod
{

namespace
{

template <typename T>
T *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
#if CAFFE2_NIGHTLY_VERSION >= 20191010
    return tensor.data_ptr<T>();
#else
    return tensor.data<T>();
#endif
}

template <>
uint16_t *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    NEUROPOD_ERROR("TorchScript doesn't support type uint16_t");
}

template <>
uint32_t *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    NEUROPOD_ERROR("TorchScript doesn't support type uint32_t");
}

template <>
uint64_t *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    NEUROPOD_ERROR("TorchScript doesn't support type uint64_t");
}

torch::Deleter get_torch_deleter(const Deleter &deleter, void *data)
{
    auto handle = register_deleter(deleter, data);
    return [handle](void *unused) { run_deleter(handle); };
}

} // namespace

// This class is internal to neuropods and should not be exposed
// to users
template <typename T>
class TorchNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<torch::jit::IValue>
{
public:
    // Allocate a torch tensor
    TorchNeuropodTensor(const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<T>(dims),
          tensor(torch::empty(dims, get_torch_type_from_neuropod_type(get_tensor_type_from_cpp<T>())))
    {
    }

    // Wrap existing memory
    TorchNeuropodTensor(const std::vector<int64_t> &dims, void *data, const Deleter &deleter)
        : TypedNeuropodTensor<T>(dims),
          tensor(torch::from_blob(data,
                                  dims,
                                  get_torch_deleter(deleter, data),
                                  get_torch_type_from_neuropod_type(get_tensor_type_from_cpp<T>())))
    {
    }

    // Wrap an existing torch tensor
    TorchNeuropodTensor(torch::Tensor tensor) : TypedNeuropodTensor<T>(tensor.sizes().vec()), tensor(tensor) {}

    ~TorchNeuropodTensor() = default;

    torch::jit::IValue get_native_data() { return tensor; }

    // The underlying torch tensor
    torch::Tensor tensor;

protected:
    // Get a pointer to the underlying data
    void *get_untyped_data_ptr() { return get_data_from_torch_tensor<T>(tensor); }

    // Get a pointer to the underlying data
    const void *get_untyped_data_ptr() const { return get_data_from_torch_tensor<T>(tensor); }
};

#if CAFFE2_NIGHTLY_VERSION >= 20190717
#define ELEMENTS(collection) collection
#define GET_STRING_FROM_LIST(item) item
#else
#define ELEMENTS(collectionptr) collectionptr->elements();
#define GET_STRING_FROM_LIST(item) item.toStringRef()
#endif

// Specialization for strings
// Torch does not natively support string tensors. Instead, we will be using a list of strings.
// Note: this only implements support for 1D string tensors
// TODO(vip, yevgeni): Design a better approach to multidimensional string tensors
template <>
class TorchNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>,
                                         public NativeDataContainer<torch::jit::IValue>
{
public:
    // Allocate a torch tensor
    TorchNeuropodTensor(const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<std::string>(dims),
#if CAFFE2_NIGHTLY_VERSION >= 20190717
          list(std::vector<std::string>(get_num_elements()))
#else
          list(at::ivalue::GenericList::create(std::vector<torch::jit::IValue>(get_num_elements())))
#endif
    {
        if (dims.size() != 1)
        {
            NEUROPOD_ERROR("Only 1D TorchScript string tensors are supported. "
                           "Tried to create a tensor with "
                           << dims.size() << " dimensions.");
        }
    }

    // Wrap an existing torch tensor
    TorchNeuropodTensor(torch::jit::IValue tensor)
        : TypedNeuropodTensor<std::string>({static_cast<int64_t>(tensor.toGenericListRef().size())}),
#if CAFFE2_NIGHTLY_VERSION >= 20190717
          list(c10::impl::toTypedList<std::string>(tensor.toGenericList()))
#else
          list(tensor.toGenericList())
#endif
    {
    }

    ~TorchNeuropodTensor() = default;

    void set(const std::vector<std::string> &data)
    {
        if (data.size() != get_num_elements())
        {
            NEUROPOD_ERROR("Error setting data for a TorchScript string tensor. "
                           "Make sure that the number of elements in the input vector is correct. "
                           "Expected size "
                           << get_num_elements() << " but got " << data.size());
        }

        // Get a reference to the tensor data
        auto &tensor_data = ELEMENTS(list);
        for (size_t i = 0; i < data.size(); i++)
        {
            // Set each item
            tensor_data[i] = data[i];
        }
    }

    std::vector<std::string> get_data_as_vector() const
    {
        std::vector<std::string> out;

        // Reserve space for all the items in the tensor
        out.reserve(get_num_elements());

        // Sanity check sizes
        auto &tensor_data = ELEMENTS(list);
        if (tensor_data.size() != get_num_elements())
        {
            NEUROPOD_ERROR("Error converting TorchScript list into vector of strings. "
                           "Make sure that the dimensions of the returned list are correct. "
                           "Expected size "
                           << get_num_elements() << " but got " << tensor_data.size());
        }

        for (const auto &item : tensor_data)
        {
            out.emplace_back(GET_STRING_FROM_LIST(item));
        }

        // Return the filled vector
        return out;
    }

#if CAFFE2_NIGHTLY_VERSION >= 20190717
    // Store a typed list
    torch::jit::IValue     get_native_data() { return c10::impl::toGenericList(list); }
    c10::List<std::string> list;
#else
    // Store a generic list
    torch::jit::IValue                          get_native_data() { return list; }
    c10::intrusive_ptr<at::ivalue::GenericList> list;
#endif
};

// Utility function to get an IValue from a torch tensor
torch::jit::IValue get_ivalue_from_torch_tensor(const std::shared_ptr<NeuropodValue> &tensor)
{
    return std::dynamic_pointer_cast<NativeDataContainer<torch::jit::IValue>>(tensor)->get_native_data();
}

} // namespace neuropod
