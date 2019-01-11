//
// Uber, Inc. (c) 2018
//

#include "torch_tensor.hh"

#include <sstream>
#include <stdexcept>

#include "neuropods/backends/torchscript/type_utils.hh"

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

// Allocate a new tensor
template <typename T>
TorchNeuropodTensor<T>::TorchNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
    : TypedNeuropodTensor<T>(name, dims),
      tensor(torch::empty(dims, get_torch_type_from_neuropod_type(get_tensor_type_from_cpp<T>())))
{
}

// Wrap an existing tensor
template <typename T>
TorchNeuropodTensor<T>::TorchNeuropodTensor(const std::string &name, torch::Tensor tensor)
    : TypedNeuropodTensor<T>(name, tensor.sizes().vec()), tensor(tensor)
{
}

template <typename T>
TorchNeuropodTensor<T>::~TorchNeuropodTensor() = default;

template <typename T>
T *TorchNeuropodTensor<T>::get_raw_data_ptr()
{
    return get_data_from_torch_tensor<T>(tensor);
}

template <typename T>
torch::jit::IValue TorchNeuropodTensor<T>::get_native_data()
{
    return tensor;
}

// Instantiate the templates
#define INIT_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE) template class TorchNeuropodTensor<CPP_TYPE>;

FOR_EACH_TYPE_MAPPING_INCLUDING_STRING(INIT_TEMPLATES_FOR_TYPE);

} // namespace neuropods
