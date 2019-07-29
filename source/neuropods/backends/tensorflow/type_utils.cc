//
// Uber, Inc. (c) 2018
//

#include "type_utils.hh"

#include "neuropods/internal/error_utils.hh"

#include <sstream>
#include <stdexcept>

namespace neuropods
{

#define FOR_TF_NEUROPOD_MAPPING(FN) \
    FN(FLOAT_TENSOR, TF_FLOAT)      \
    FN(DOUBLE_TENSOR, TF_DOUBLE)    \
    FN(STRING_TENSOR, TF_STRING)    \
                                    \
    FN(INT8_TENSOR, TF_INT8)        \
    FN(INT16_TENSOR, TF_INT16)      \
    FN(INT32_TENSOR, TF_INT32)      \
    FN(INT64_TENSOR, TF_INT64)      \
                                    \
    FN(UINT8_TENSOR, TF_UINT8)      \
    FN(UINT16_TENSOR, TF_UINT16)    \
    FN(UINT32_TENSOR, TF_UINT32)    \
    FN(UINT64_TENSOR, TF_UINT64)

TensorType get_neuropod_type_from_tf_type(TF_DataType type)
{
#define TF_TO_NEUROPOD(NEUROPOD_TYPE, TF_TYPE) \
    case TF_TYPE:                              \
        return NEUROPOD_TYPE;

    switch (type)
    {
        FOR_TF_NEUROPOD_MAPPING(TF_TO_NEUROPOD)
    default:
        break;
    }

    NEUROPOD_ERROR("Neuropods does not support type: " << type);
}

TF_DataType get_tf_type_from_neuropod_type(TensorType type)
{
#define NEUROPOD_TO_TF(NEUROPOD_TYPE, TF_TYPE) \
    case NEUROPOD_TYPE:                        \
        return TF_TYPE;

    switch (type)
    {
        FOR_TF_NEUROPOD_MAPPING(NEUROPOD_TO_TF)
    }

    NEUROPOD_ERROR("TensorFlow does not support type: " << type);
}

} // namespace neuropods
