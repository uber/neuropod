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

#include "neuropod/bindings/c/np_tensor_internal.h"
#include "neuropod/bindings/c/np_valuemap_internal.h"

// Create a new NeuropodValueMap
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
NP_NeuropodValueMap *NP_NewValueMap()
{
    return new NP_NeuropodValueMap();
}

// Free a NeuropodValueMap
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_FreeValueMap(NP_NeuropodValueMap *nvm)
{
    delete nvm;
}

// Insert a tensor into the provided map
// Overwrites any existing tensors with the same name
// Note: this does not transfer ownership and the caller is still responsible for calling NP_FreeTensor on
// `tensor`
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_InsertTensor(NP_NeuropodValueMap *nvm, const char *name, NP_NeuropodTensor *tensor)
{
    nvm->data[name] = tensor->tensor;
}

// Given a NP_NeuropodValueMap, find a tensor with name `name` and return it
// If such a tensor is not found, returns nullptr
// Note: the caller is responsible for calling NP_FreeTensor on the returned tensor
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
NP_NeuropodTensor *NP_GetTensor(const NP_NeuropodValueMap *nvm, const char *name)
{
    auto &map  = nvm->data;
    auto  item = map.find(name);
    if (item == map.end())
    {
        return nullptr;
    }

    auto retval    = new NP_NeuropodTensor();
    retval->tensor = item->second;

    return retval;
}

// Removes a specified tensor from the provided map (if it exists)
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_RemoveTensor(NP_NeuropodValueMap *nvm, const char *name)
{
    nvm->data.erase(name);
}
