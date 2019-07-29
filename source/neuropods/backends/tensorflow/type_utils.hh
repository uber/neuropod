//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/internal/tensor_types.hh"

#include <tensorflow/c/c_api.h>

namespace neuropods
{

TensorType  get_neuropod_type_from_tf_type(TF_DataType type);
TF_DataType get_tf_type_from_neuropod_type(TensorType type);

} // namespace neuropods
