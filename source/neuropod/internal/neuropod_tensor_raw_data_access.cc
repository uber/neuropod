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

#include "neuropod/internal/neuropod_tensor_raw_data_access.hh"

#include "neuropod/internal/neuropod_tensor.hh"

namespace neuropod::internal
{

void *NeuropodTensorRawDataAccess::get_untyped_data_ptr(NeuropodTensor &tensor)
{
    return tensor.get_untyped_data_ptr();
}

const void *NeuropodTensorRawDataAccess::get_untyped_data_ptr(const NeuropodTensor &tensor)
{
    return tensor.get_untyped_data_ptr();
}

size_t NeuropodTensorRawDataAccess::get_bytes_per_element(const NeuropodTensor &tensor)
{
    return tensor.get_bytes_per_element();
}

} // namespace neuropod::internal
