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

#include "neuropod/internal/tensor_types.hh"

namespace neuropod
{

// Used to print out the enum names rather than just a number
std::ostream &operator<<(std::ostream &out, const TensorType value)
{
    const char *s = nullptr;
#define GENERATE_CASE(item) \
    case (item):            \
        s = #item;          \
        break
    switch (value)
    {
        GENERATE_CASE(FLOAT_TENSOR);
        GENERATE_CASE(DOUBLE_TENSOR);
        GENERATE_CASE(STRING_TENSOR);
        GENERATE_CASE(INT8_TENSOR);
        GENERATE_CASE(INT16_TENSOR);
        GENERATE_CASE(INT32_TENSOR);
        GENERATE_CASE(INT64_TENSOR);
        GENERATE_CASE(UINT8_TENSOR);
        GENERATE_CASE(UINT16_TENSOR);
        GENERATE_CASE(UINT32_TENSOR);
        GENERATE_CASE(UINT64_TENSOR);
    }
#undef GENERATE_CASE

    return out << s;
}

} // namespace neuropod
