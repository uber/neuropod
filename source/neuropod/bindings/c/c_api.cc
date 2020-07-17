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

#include "neuropod/bindings/c/c_api.h"

#include "neuropod/bindings/c/c_api_internal.h"
#include "neuropod/bindings/c/np_tensor_allocator_internal.h"
#include "neuropod/bindings/c/np_valuemap_internal.h"

// Load a model given a path
void NP_LoadNeuropod(const char *neuropod_path, NP_Neuropod **model, NP_Status *status)
{
    *model = new NP_Neuropod();

    (*model)->model = std::make_unique<neuropod::Neuropod>(neuropod_path);
}

// Free a model
void NP_FreeNeuropod(NP_Neuropod *model)
{
    delete model;
}

// Run inference
// Note: The caller is responsible for freeing the returned NP_NeuropodValueMap
void NP_Infer(NP_Neuropod *model, const NP_NeuropodValueMap *inputs, NP_NeuropodValueMap **outputs, NP_Status *status)
{
    *outputs         = new NP_NeuropodValueMap();
    (*outputs)->data = std::move(*model->model->infer(inputs->data));
}

// Get an allocator for a model
NP_TensorAllocator *NP_GetAllocator(NP_Neuropod *model)
{
    auto out       = new NP_TensorAllocator();
    out->allocator = model->model->get_tensor_allocator();
    return out;
}
