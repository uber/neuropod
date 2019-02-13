//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

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
    throw std::runtime_error("String support is not implemented yet");
}

template <>
uint16_t *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    throw std::runtime_error("TorchScript doesn't support type uint16_t");
}

template <>
uint32_t *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    throw std::runtime_error("TorchScript doesn't support type uint32_t");
}

template <>
uint64_t *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    throw std::runtime_error("TorchScript doesn't support type uint64_t");
}

} // namespace

// This class is internal to neuropods and should not be exposed
// to users
template <typename T>
class TorchNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<torch::jit::IValue>
{
public:
    // Allocate a torch tensor
    // TODO(vip): maybe add a way to wrap existing data using torch::from_blob
    TorchNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<T>(name, dims),
          tensor(torch::empty(dims, get_torch_type_from_neuropod_type(get_tensor_type_from_cpp<T>())))
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

} // namespace neuropods
