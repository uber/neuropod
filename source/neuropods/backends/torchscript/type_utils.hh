//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/internal/tensor_types.hh"

#include <torch/torch.h>

namespace neuropods
{

TensorType   get_neuropod_type_from_torch_type(torch::Dtype type);
torch::Dtype get_torch_type_from_neuropod_type(TensorType type);

} // namespace neuropods
