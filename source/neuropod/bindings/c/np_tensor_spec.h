/* Copyright (c) 2020 UATC, LLC

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

// Inspired by the TensorFlow C API

#pragma once

#include "neuropod/bindings/c/np_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NP_Dimension
{
    // The value of the dimension
    //
    // Special cases
    // -1 == Any value is allowed (None/null)
    // -2 == Symbol (see below)
    int64_t value;

    // The name of this symbol (if it is a symbol)
    const char *symbol;
} NP_Dimension;

typedef struct NP_TensorSpec
{
    // The name of the tensor
    const char *name;

    // The type of the tensor
    NP_TensorType type;

    // The number of dimensions
    size_t num_dims;

    // The dimensions of the tensor
    NP_Dimension dims[];
} NP_TensorSpec;

// Free a NP_TensorSpec
void NP_FreeTensorSpec(NP_TensorSpec *spec);

#ifdef __cplusplus
}
#endif
