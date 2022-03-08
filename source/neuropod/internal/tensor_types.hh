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
