//
// Uber, Inc. (c) 2018
//

#pragma once

#include <ostream>

namespace neuropod
{

// All the tensor types that Neuropod supports
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

// Used to print out the enum names rather than just a number
std::ostream &operator<<(std::ostream &out, const TensorType value);

} // namespace neuropod
