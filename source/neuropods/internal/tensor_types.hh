//
// Uber, Inc. (c) 2018
//

#pragma once

namespace neuropods
{

// All the tensor types that Neuropods supports
enum TensorType
{
    FLOAT_TENSOR,
    DOUBLE_TENSOR,
    STRING_TENSOR,

    INT8_TENSOR,
    INT16_TENSOR,
    INT32_TENSOR,
    INT64_TENSOR,

    UINT8_TENSOR,
    UINT16_TENSOR,
    UINT32_TENSOR,
    UINT64_TENSOR,
};

} // namespace neuropods
