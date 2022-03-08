/* Copyright (c) 2020 The Neuropod Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "type_utils.hh"

#include "neuropod/internal/error_utils.hh"

#include <sstream>
#include <stdexcept>

namespace neuropod
{

#define FOR_TF_NEUROPOD_MAPPING(FN)          \
    FN(FLOAT_TENSOR, tensorflow::DT_FLOAT)   \
    FN(DOUBLE_TENSOR, tensorflow::DT_DOUBLE) \
    FN(STRING_TENSOR, tensorflow::DT_STRING) \
                                             \
    FN(INT8_TENSOR, tensorflow::DT_INT8)     \
    FN(INT16_TENSOR, tensorflow::DT_INT16)   \
    FN(INT32_TENSOR, tensorflow::DT_INT32)   \
    FN(INT64_TENSOR, tensorflow::DT_INT64)   \
                                             \
    FN(UINT8_TENSOR, tensorflow::DT_UINT8)   \
    FN(UINT16_TENSOR, tensorflow::DT_UINT16) \
    FN(UINT32_TENSOR, tensorflow::DT_UINT32) \
    FN(UINT64_TENSOR, tensorflow::DT_UINT64)

TensorType get_neuropod_type_from_tf_type(tensorflow::DataType type)
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

    NEUROPOD_ERROR("Neuropod does not support type: {}", type);
}

tensorflow::DataType get_tf_type_from_neuropod_type(TensorType type)
{
#define NEUROPOD_TO_TF(NEUROPOD_TYPE, TF_TYPE) \
    case NEUROPOD_TYPE:                        \
        return TF_TYPE;

    switch (type)
    {
        FOR_TF_NEUROPOD_MAPPING(NEUROPOD_TO_TF)
    }

    NEUROPOD_ERROR("TensorFlow does not support type: {}", type);
}

} // namespace neuropod
