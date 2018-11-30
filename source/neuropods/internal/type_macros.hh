//
// Uber, Inc. (c) 2018
//

#pragma once

#include "tensor_types.hh"

#include <string>

// Macros for when we need to do something for each supported type
// UATG(clang-format/format) intentionally formatted to make types clear
#define FOR_EACH_TYPE_MAPPING_DELIM(FN, DELIM) \
    FN(float, FLOAT_TENSOR)             DELIM \
    FN(double, DOUBLE_TENSOR)           DELIM \
    FN(std::string, STRING_TENSOR)      DELIM \
                                              \
    FN(int8_t, INT8_TENSOR)             DELIM \
    FN(int16_t, INT16_TENSOR)           DELIM \
    FN(int32_t, INT32_TENSOR)           DELIM \
    FN(int64_t, INT64_TENSOR)           DELIM \
                                              \
    FN(uint8_t, UINT8_TENSOR)           DELIM \
    FN(uint16_t, UINT16_TENSOR)         DELIM \
    FN(uint32_t, UINT32_TENSOR)         DELIM \
    FN(uint64_t, UINT64_TENSOR)


#define FOR_EACH_TYPE_MAPPING(FN) FOR_EACH_TYPE_MAPPING_DELIM(FN, NO_DELIM)

#define COMMA_DELIM ,
#define NO_DELIM
