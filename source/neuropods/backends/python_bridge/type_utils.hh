//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/internal/tensor_types.hh"

namespace neuropods
{

int        get_numpy_type_from_neuropod_type(TensorType type);
TensorType get_neuropod_type_from_numpy_type(int type);

} // namespace neuropods
