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

#include <cstdlib>
#include <string>
#include <vector>

namespace neuropod
{

class NeuropodTensor;

template <typename T>
class TypedNeuropodTensor;

namespace internal
{

// This struct is used internally within the library to access raw untyped data
// from a NeuropodTensor
//
// This should NOT be used externally
struct NeuropodTensorRawDataAccess
{
    static void *get_untyped_data_ptr(NeuropodTensor &tensor);

    static const void *get_untyped_data_ptr(const NeuropodTensor &tensor);

    static size_t get_bytes_per_element(const NeuropodTensor &tensor);
};

} // namespace internal
} // namespace neuropod
