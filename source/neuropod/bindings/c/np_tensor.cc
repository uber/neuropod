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

#include "neuropod/bindings/c/np_tensor.h"

#include "neuropod/bindings/c/np_tensor_internal.h"
#include "neuropod/internal/neuropod_tensor_raw_data_access.hh"

// For non-string tensors, get a pointer to the underlying data
// Returns nullptr if called on a string tensor
void *NP_GetData(NP_NeuropodTensor *tensor)
{
    return neuropod::internal::NeuropodTensorRawDataAccess::get_untyped_data_ptr(*tensor->tensor->as_tensor());
}

// Releases a tensor. The memory might not be deallocated immediately if the tensor is still
// referenced by a value-map object or used by an infer operation.
// This should be called on every tensor returned by the C API.
// See the notes in `np_valuemap.h` and `np_tensor_allocator.h` for more detail.
void NP_FreeTensor(NP_NeuropodTensor *tensor)
{
    delete tensor;
}
