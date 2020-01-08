//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropod/internal/tensor_types.hh"
#include "tensorflow/core/framework/types.pb.h"

namespace neuropod
{

TensorType           get_neuropod_type_from_tf_type(tensorflow::DataType type);
tensorflow::DataType get_tf_type_from_neuropod_type(TensorType type);

} // namespace neuropod
