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

typedef struct NP_TensorAllocator NP_TensorAllocator;

// Free an allocator
void NP_FreeAllocator(NP_TensorAllocator *allocator);

// Allocate a tensor with a specified type and shape
// Note: the caller is responsible for calling NP_FreeTensor on the returned tensor
NP_NeuropodTensor *NP_AllocateTensor(NP_TensorAllocator *allocator, size_t num_dims, int64_t *dims, NP_TensorType type);

// Allocate a tensor of a specific type and wrap existing memory.
// Note: Some backends may have specific alignment requirements (e.g. tensorflow).
// To support all the built-in backends, `data` should be aligned to 64 bytes.
// `deleter` will be called with a pointer to `data` and `deleter_arg` when the tensor is
// deallocated.
// Note: the caller is responsible for calling NP_FreeTensor on the returned tensor
NP_NeuropodTensor *NP_TensorFromMemory(NP_TensorAllocator *allocator,
                                       size_t              num_dims,
                                       int64_t *           dims,
                                       NP_TensorType       type,
                                       void *              data,
                                       void (*deleter)(void *data, void *arg),
                                       void *deleter_arg);

#ifdef __cplusplus
}
#endif
