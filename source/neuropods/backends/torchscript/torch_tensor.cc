//
// Uber, Inc. (c) 2018
//

#include "torch_tensor.hh"

#include <sstream>
#include <stdexcept>

namespace neuropods
{

namespace
{

#define FOR_TORCH_NEUROPOD_MAPPING(FN) \
    FN(FLOAT_TENSOR, torch::kFloat32)  \
    FN(DOUBLE_TENSOR, torch::kFloat64) \
                                       \
    FN(INT8_TENSOR, torch::kInt8)      \
    FN(INT16_TENSOR, torch::kInt16)    \
    FN(INT32_TENSOR, torch::kInt32)    \
    FN(INT64_TENSOR, torch::kInt64)    \
                                       \
    FN(UINT8_TENSOR, torch::kUInt8)    \
    // TODO(vip): add string support
    // FN(STRING_TENSOR, ...)
    //
    // Unsupported types:
    // FN(UINT16_TENSOR, ...)
    // FN(UINT32_TENSOR, ...)
    // FN(UINT64_TENSOR, ...)


TensorType get_neuropod_type_from_torch_type(torch::Dtype type)
{
#define TORCH_TO_NEUROPOD(NEUROPOD_TYPE, TORCH_TYPE) \
    case TORCH_TYPE:                                 \
        return NEUROPOD_TYPE;

    switch (type)
    {
        FOR_TORCH_NEUROPOD_MAPPING(TORCH_TO_NEUROPOD)
    default:
        break;
    }

    std::stringstream ss;
    ss << "Neuropods does not support type: " << type;
    throw std::runtime_error(ss.str());
}

torch::Dtype get_torch_type_from_neuropod_type(TensorType type)
{
#define NEUROPOD_TO_TORCH(NEUROPOD_TYPE, TORCH_TYPE) \
    case NEUROPOD_TYPE:                              \
        return TORCH_TYPE;

    switch (type)
    {
        FOR_TORCH_NEUROPOD_MAPPING(NEUROPOD_TO_TORCH)
    default:
        break;
    }

    std::stringstream ss;
    ss << "TorchScript does not support type: " << type;
    throw std::runtime_error(ss.str());
}

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
TorchNeuropodTensor::TorchNeuropodTensor(const std::string &         name,
                                         const std::vector<int64_t> &dims,
                                         TensorType                  tensor_type)
    : NeuropodTensor(name, tensor_type, dims),
      tensor(torch::empty(dims, get_torch_type_from_neuropod_type(tensor_type)))
{
}

// Wrap an existing tensor
TorchNeuropodTensor::TorchNeuropodTensor(const std::string &name, torch::Tensor tensor)
    : NeuropodTensor(name, get_neuropod_type_from_torch_type(tensor.scalar_type()), tensor.sizes().vec()),
      tensor(tensor)
{
}

TorchNeuropodTensor::~TorchNeuropodTensor() = default;

TensorDataPointer TorchNeuropodTensor::get_data_ptr()
{
#define CAST_TENSOR(CPP_TYPE, NEUROPOD_TYPE)                 \
    case NEUROPOD_TYPE:                                      \
    {                                                        \
        return get_data_from_torch_tensor<CPP_TYPE>(tensor); \
    }

    // Cast it to the correct type and return
    switch (get_tensor_type())
    {
        FOR_EACH_TYPE_MAPPING(CAST_TENSOR)
    }
}

} // namespace neuropods
