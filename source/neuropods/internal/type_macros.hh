//
// Uber, Inc. (c) 2018
//

#pragma once

#include "tensor_types.hh"

#include <string>

// Macros for when we need to do something for each supported type
// Note: strings are special and are handled separately
#define FOR_EACH_TYPE_MAPPING_EXCEPT_STRING(FN) \
    FN(float, FLOAT_TENSOR)                     \
    FN(double, DOUBLE_TENSOR)                   \
                                                \
    FN(int8_t, INT8_TENSOR)                     \
    FN(int16_t, INT16_TENSOR)                   \
    FN(int32_t, INT32_TENSOR)                   \
    FN(int64_t, INT64_TENSOR)                   \
                                                \
    FN(uint8_t, UINT8_TENSOR)                   \
    FN(uint16_t, UINT16_TENSOR)                 \
    FN(uint32_t, UINT32_TENSOR)                 \
    FN(uint64_t, UINT64_TENSOR)

#define FOR_EACH_TYPE_MAPPING_INCLUDING_STRING(FN) \
    FOR_EACH_TYPE_MAPPING_EXCEPT_STRING(FN)        \
    FN(std::string, STRING_TENSOR)
