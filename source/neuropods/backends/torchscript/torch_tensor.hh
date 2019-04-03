//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"

namespace neuropods
{

namespace
{

template <typename T>
T *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    return tensor.data<T>();
}

template <>
std::string *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    NEUROPOD_ERROR("String support is not implemented yet");
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

torch::Deleter get_torch_deleter(const Deleter &deleter, void * data)
{
    auto handle = register_deleter(deleter, data);
    return [handle](void * unused) {
        run_deleter(handle);
    };
}

} // namespace

// This class is internal to neuropods and should not be exposed
// to users
template <typename T>
class TorchNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<torch::jit::IValue>
{
public:
    // Allocate a torch tensor
    TorchNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<T>(name, dims),
          tensor(torch::empty(dims, get_torch_type_from_neuropod_type(get_tensor_type_from_cpp<T>())))
    {
    }

    // Wrap existing memory
    TorchNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims, void * data, const Deleter &deleter)
        : TypedNeuropodTensor<T>(name, dims),
          tensor(torch::from_blob(data, dims, get_torch_deleter(deleter, data), get_torch_type_from_neuropod_type(get_tensor_type_from_cpp<T>())))
    {
    }

    // Wrap an existing torch tensor
    TorchNeuropodTensor(const std::string &name, torch::Tensor tensor)
        : TypedNeuropodTensor<T>(name, tensor.sizes().vec()), tensor(tensor)
    {
    }

    ~TorchNeuropodTensor() = default;

    // Get a pointer to the underlying data
    T *get_raw_data_ptr() { return get_data_from_torch_tensor<T>(tensor); }

    // Get a pointer to the underlying data
    const T *get_raw_data_ptr() const { return get_data_from_torch_tensor<T>(tensor); }

    torch::jit::IValue get_native_data() { return tensor; }

    // The underlying torch tensor
    torch::Tensor tensor;
};

// Specialization for strings
// Torch does not natively support string tensors. Instead, we will be using a list of strings.
// Note: this only implements support for 1D string tensors
// TODO(vip, yevgeni): Design a better approach to multidimensional string tensors
template<>
class TorchNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>,
                                         public NativeDataContainer<torch::jit::IValue>
{
public:
    // Allocate a torch tensor
    TorchNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<std::string>(name, dims),
          list(at::ivalue::GenericList::create(std::vector<torch::jit::IValue>(get_num_elements())))
    {
        if (dims.size() != 1)
        {
            NEUROPOD_ERROR("Only 1D TorchScript string tensors are supported. "
                "Tried to create a tensor with " << dims.size() << " dimensions.");
        }
    }

    // Wrap an existing torch tensor
    TorchNeuropodTensor(const std::string &name, torch::jit::IValue tensor)
        : TypedNeuropodTensor<std::string>(name, {tensor.toGenericListRef().size()}),
          list(tensor.toGenericList())
    {
    }

    ~TorchNeuropodTensor() = default;

    void set(const std::vector<std::string> &data)
    {
        if (data.size() != get_num_elements())
        {
            NEUROPOD_ERROR("Error setting data for a TorchScript string tensor. "
                "Make sure that the number of elements in the input vector is correct. "
                "Expected size " <<  get_num_elements() << " but got " << data.size());
        }

        // Get a reference to the tensor data
        auto &tensor_data = list->elements();
        for (size_t i = 0; i < data.size(); i++)
        {
            // Wrap all the input strings with IValues and set each item
            tensor_data[i] = torch::jit::IValue(data[i]);
        }
    }

    std::vector<std::string> get_data_as_vector()
    {
        std::vector<std::string> out;

        // Reserve space for all the items in the tensor
        out.reserve(get_num_elements());

        // Get the items in the tensor and sanity check sizes
        auto &tensor_data = list->elements();
        if (tensor_data.size() != get_num_elements())
        {
            NEUROPOD_ERROR("Error converting TorchScript list into vector of strings. "
                "Make sure that the dimensions of the returned list are correct. "
                "Expected size " <<  get_num_elements() << " but got " << tensor_data.size());
        }

        for (const auto &item : tensor_data)
        {
            out.emplace_back(item.toStringRef());
        }

         // Return the filled vector
        return out;
    }

    torch::jit::IValue get_native_data() { return list; }

    // The underlying TorchScript list
    c10::intrusive_ptr<at::ivalue::GenericList> list;
};

// Utility function to get an IValue from a torch tensor
torch::jit::IValue get_ivalue_from_torch_tensor(const std::shared_ptr<NeuropodTensor> &tensor)
{
    return std::dynamic_pointer_cast<NativeDataContainer<torch::jit::IValue>>(tensor)->get_native_data();
}

} // namespace neuropods
