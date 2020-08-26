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

void NP_GetDims(const NP_NeuropodTensor *tensor, size_t *num_dims, const int64_t **dims)
{
    // get_dims return refernce to internal tensor's vector.
    // We return a pointer to it that is valid while tensor is valid.
    auto &dims_collection = tensor->tensor->as_tensor()->get_dims();
    *num_dims             = dims_collection.size();
    *dims                 = dims_collection.data();
}

NP_TensorType NP_GetType(const NP_NeuropodTensor *tensor)
{
    return static_cast<NP_TensorType>(tensor->tensor->as_tensor()->get_tensor_type());
}

void *NP_GetData(NP_NeuropodTensor *tensor)
{
    return neuropod::internal::NeuropodTensorRawDataAccess::get_untyped_data_ptr(*tensor->tensor->as_tensor());
}

const void *NP_GetDataReadOnly(const NP_NeuropodTensor *tensor)
{
    return neuropod::internal::NeuropodTensorRawDataAccess::get_untyped_data_ptr(*tensor->tensor->as_tensor());
}

size_t NP_GetNumElements(const NP_NeuropodTensor *tensor)
{
    return tensor->tensor->as_tensor()->get_num_elements();
}

void NP_FreeTensor(NP_NeuropodTensor *tensor)
{
    delete tensor;
}
