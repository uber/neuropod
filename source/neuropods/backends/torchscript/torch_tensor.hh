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

// This class is internal to neuropods and should not be exposed
// to users
template <typename T>
class TorchNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<torch::jit::IValue>
{
public:
    // Allocate a torch tensor
    // TODO(vip): maybe add a way to wrap existing data using torch::from_blob
    TorchNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims);

    // Wrap an existing torch tensor
    TorchNeuropodTensor(const std::string &name, torch::Tensor tensor);

    ~TorchNeuropodTensor();

    // Get a pointer to the underlying data
    T *get_raw_data_ptr();

    torch::jit::IValue get_native_data();

    // The underlying torch tensor
    torch::Tensor tensor;
};

} // namespace neuropods
