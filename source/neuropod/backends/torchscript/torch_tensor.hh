/* Copyright (c) 2020 UATC, LLC

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

#pragma once

#include "neuropod/internal/deleter.hh"
#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/neuropod_tensor.hh"

#include <caffe2/core/macros.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>

// If we're not building with a nightly relase of torch,
// set the date to match the date of the official release
#ifndef CAFFE2_NIGHTLY_VERSION
#if CAFFE2_VERSION == 10100
// The date of the official torch 1.1.0 release
#define CAFFE2_NIGHTLY_VERSION 20190430
#endif

#if CAFFE2_VERSION == 10200
// The date of the official torch 1.2.0 release
#define CAFFE2_NIGHTLY_VERSION 20190808
#endif

#if CAFFE2_VERSION == 10300
// The date of the official torch 1.3.0 release
#define CAFFE2_NIGHTLY_VERSION 20191010
#endif

#if CAFFE2_VERSION == 10400
// The date of the official torch 1.4.0 release
#define CAFFE2_NIGHTLY_VERSION 20200115
#endif

#if CAFFE2_VERSION == 10500
// The date of the official torch 1.5.0 release
#define CAFFE2_NIGHTLY_VERSION 20200421
#endif

#if CAFFE2_VERSION == 10600
// The date of the official torch 1.6.0 release
#define CAFFE2_NIGHTLY_VERSION 20200728
#endif

#if CAFFE2_VERSION == 10700
// The date of the official torch 1.7.0 release
#define CAFFE2_NIGHTLY_VERSION 20201027
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
[[maybe_unused]] uint16_t *get_data_from_torch_tensor(const torch::Tensor & /*unused*/)
{
    NEUROPOD_ERROR("TorchScript doesn't support type uint16_t");
}

template <>
[[maybe_unused]] uint32_t *get_data_from_torch_tensor(const torch::Tensor & /*unused*/)
{
    NEUROPOD_ERROR("TorchScript doesn't support type uint32_t");
}

template <>
[[maybe_unused]] uint64_t *get_data_from_torch_tensor(const torch::Tensor & /*unused*/)
{
    NEUROPOD_ERROR("TorchScript doesn't support type uint64_t");
}

[[maybe_unused]] torch::Deleter get_torch_deleter(const Deleter &deleter, void *data)
{
    auto handle = register_deleter(deleter, data);
    return [handle](void * /*unused*/) { run_deleter(handle); };
}

} // namespace

// This class is internal to neuropod and should not be exposed
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
    TorchNeuropodTensor(torch::Tensor t) : TypedNeuropodTensor<T>(t.sizes().vec()), tensor(t) {}

    ~TorchNeuropodTensor() = default;

    torch::jit::IValue get_native_data() { return tensor; }

    // The underlying torch tensor
    torch::Tensor tensor;

protected:
    // Get a pointer to the underlying data
    void *get_untyped_data_ptr() { return get_data_from_torch_tensor<T>(tensor); }

    // Get a pointer to the underlying data
    const void *get_untyped_data_ptr() const { return get_data_from_torch_tensor<T>(tensor); }

    std::shared_ptr<NeuropodValue> to_internal(NeuropodDevice device)
    {
        torch::Tensor out;
        if (device != Device::CPU && !torch::cuda::is_available())
        {
            // No matter what the target device is, we don't have a choice other than running on CPU
            SPDLOG_WARN("Tried to move a torch tensor to GPU, but CUDA isn't available. Falling back to CPU");
            out = tensor.to(torch::kCPU);
        }
        else if (device == Device::CPU)
        {
            out = tensor.to(torch::kCPU);
        }
        else
        {
            out = tensor.to(torch::Device(torch::kCUDA, static_cast<short>(device)));
        }

        return std::make_shared<TorchNeuropodTensor<T>>(std::move(out));
    }
};

#if CAFFE2_NIGHTLY_VERSION >= 20190717
#define ELEMENTS(collection) collection
#define GET_STRING_FROM_LIST(item) item
#else
#define ELEMENTS(collectionptr) collectionptr->elements()
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
                           "Tried to create a tensor with {} dimensions.",
                           dims.size());
        }
    }

    // Wrap an existing torch tensor
    TorchNeuropodTensor(torch::jit::IValue tensor)
#if CAFFE2_NIGHTLY_VERSION >= 20200421
        : TypedNeuropodTensor<std::string>({static_cast<int64_t>(tensor.toListRef().size())}),
          list(c10::impl::toTypedList<std::string>(tensor.toList()))
#elif CAFFE2_NIGHTLY_VERSION >= 20190717
        : TypedNeuropodTensor<std::string>({static_cast<int64_t>(tensor.toGenericListRef().size())}),
          list(c10::impl::toTypedList<std::string>(tensor.toGenericList()))
#else
        : TypedNeuropodTensor<std::string>({static_cast<int64_t>(tensor.toGenericListRef().size())}),
          list(tensor.toGenericList())
#endif
    {
    }

    ~TorchNeuropodTensor() = default;

    void copy_from(const std::vector<std::string> &data)
    {
        if (data.size() != get_num_elements())
        {
            NEUROPOD_ERROR("Error setting data for a TorchScript string tensor. "
                           "Make sure that the number of elements in the input vector is correct. "
                           "Expected size {} but got {}",
                           get_num_elements(),
                           data.size());
        }

        // Get a reference to the tensor data
        auto &tensor_data = ELEMENTS(list);
        for (size_t i = 0; i < data.size(); i++)
        {
            // Set each item
            tensor_data[i] = data[i];
        }
    }
#if CAFFE2_NIGHTLY_VERSION >= 20200421
    // Store a typed list
    torch::jit::IValue     get_native_data() { return c10::impl::toList(list); }
    c10::List<std::string> list;

#elif CAFFE2_NIGHTLY_VERSION >= 20190717
    // Store a typed list
    torch::jit::IValue     get_native_data() { return c10::impl::toGenericList(list); }
    c10::List<std::string> list;
#else
    // Store a generic list
    torch::jit::IValue                          get_native_data() { return list; }
    c10::intrusive_ptr<at::ivalue::GenericList> list;
#endif

protected:
    std::string get(size_t index) const
    {
        auto &tensor_data = ELEMENTS(list);
        return GET_STRING_FROM_LIST(tensor_data[index]);
    }

    void set(size_t index, const std::string &value)
    {
        auto &tensor_data  = ELEMENTS(list);
        tensor_data[index] = value;
    }
};

// Utility function to get an IValue from a torch tensor
inline torch::jit::IValue get_ivalue_from_torch_tensor(const std::shared_ptr<NeuropodValue> &tensor)
{
    return std::dynamic_pointer_cast<NativeDataContainer<torch::jit::IValue>>(tensor)->get_native_data();
}

} // namespace neuropod
