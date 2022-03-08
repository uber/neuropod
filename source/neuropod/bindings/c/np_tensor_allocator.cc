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

#include "neuropod/bindings/c/np_tensor_allocator_internal.h"
#include "neuropod/bindings/c/np_tensor_internal.h"

// Free an allocator
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_FreeAllocator(NP_TensorAllocator *allocator)
{
    delete allocator;
}

// Allocate a tensor with a specified type and shape
// Note: the caller is responsible for calling NP_FreeTensor on the returned tensor
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
NP_NeuropodTensor *NP_AllocateTensor(NP_TensorAllocator *allocator, size_t num_dims, int64_t *dims, NP_TensorType type)
{
    std::vector<int64_t> d(dims, dims + num_dims);

    auto out    = new NP_NeuropodTensor();
    out->tensor = allocator->allocator->allocate_tensor(d, static_cast<neuropod::TensorType>(type));

    return out;
}
