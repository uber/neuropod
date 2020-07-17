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

#include "neuropod/bindings/c/np_status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A tensor
typedef struct NP_NeuropodTensor NP_NeuropodTensor;

// Get the dimensions of a tensor
void NP_GetDims(const NP_NeuropodTensor *tensor, size_t *num_dims, const int64_t **dims);

// All the tensor types that Neuropod supports
typedef enum NP_TensorType
{
    FLOAT_TENSOR = 0,
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
} NP_TensorType;

// Get the type of a tensor
NP_TensorType NP_GetType(const NP_NeuropodTensor *tensor);

// For non-string tensors, get a pointer to the underlying data
// Returns nullptr if called on a string tensor
void *NP_GetData(NP_NeuropodTensor *tensor);

// For non-string tensors, get a pointer to the underlying data
// Returns nullptr if called on a string tensor
const void *NP_GetDataReadOnly(const NP_NeuropodTensor *tensor);

// For string tensors, set the value of a specified element in the flattened tensor
void NP_SetStringElement(NP_NeuropodTensor *tensor, size_t index, const char *item, NP_Status *status);

// Get the length of a specified element in a string tensor
void NP_GetStringElementSize(const NP_NeuropodTensor *tensor, size_t index, size_t *elem_size, NP_Status *status);

// Copy the value of a specified element of a string tensor into a user provided buffer
void NP_GetStringElement(
    const NP_NeuropodTensor *tensor, size_t index, char *buffer, size_t buffer_size, NP_Status *status);

// Releases a tensor. The memory might not be deallocated immediately if the tensor is still
// referenced by a value-map object or used by an infer operation.
// This should be called on every tensor returned by the C API.
// See the notes in `np_valuemap.h` and `np_tensor_allocator.h` for more detail.
void NP_FreeTensor(NP_NeuropodTensor *tensor);

#ifdef __cplusplus
}
#endif
