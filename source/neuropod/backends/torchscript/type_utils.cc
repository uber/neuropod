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

#define FOR_TORCH_NEUROPOD_MAPPING(FN) \
    FN(FLOAT_TENSOR, torch::kFloat32)  \
    FN(DOUBLE_TENSOR, torch::kFloat64) \
                                       \
    FN(INT8_TENSOR, torch::kInt8)      \
    FN(INT16_TENSOR, torch::kInt16)    \
    FN(INT32_TENSOR, torch::kInt32)    \
    FN(INT64_TENSOR, torch::kInt64)    \
                                       \
    FN(UINT8_TENSOR, torch::kUInt8)    \
    // TODO(vip): add string support
    // FN(STRING_TENSOR, ...)
    //
    // Unsupported types:
    // FN(UINT16_TENSOR, ...)
    // FN(UINT32_TENSOR, ...)
    // FN(UINT64_TENSOR, ...)

TensorType get_neuropod_type_from_torch_type(torch::Dtype type)
{
#define TORCH_TO_NEUROPOD(NEUROPOD_TYPE, TORCH_TYPE) \
    case TORCH_TYPE:                                 \
        return NEUROPOD_TYPE;

    switch (type)
    {
        FOR_TORCH_NEUROPOD_MAPPING(TORCH_TO_NEUROPOD)
    default:
        break;
    }

    NEUROPOD_ERROR("Neuropod does not support type: {}", type);
}

torch::Dtype get_torch_type_from_neuropod_type(TensorType type)
{
#define NEUROPOD_TO_TORCH(NEUROPOD_TYPE, TORCH_TYPE) \
    case NEUROPOD_TYPE:                              \
        return TORCH_TYPE;

    switch (type)
    {
        FOR_TORCH_NEUROPOD_MAPPING(NEUROPOD_TO_TORCH)
    default:
        break;
    }

    NEUROPOD_ERROR("TorchScript does not support type: {}", type);
}

} // namespace neuropod
