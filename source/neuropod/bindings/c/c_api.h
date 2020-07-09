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
#include "neuropod/bindings/c/np_tensor.h"
#include "neuropod/bindings/c/np_tensor_allocator.h"
#include "neuropod/bindings/c/np_tensor_spec.h"
#include "neuropod/bindings/c/np_valuemap.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NP_Neuropod NP_Neuropod;

// Load a model given a path
void NP_LoadNeuropod(const char *neuropod_path, NP_Neuropod **model, NP_Status *status);

// TODO: Add more options
typedef struct NP_RuntimeOptions
{
    // Whether or not to use out-of-process execution
    // (using shared memory to communicate between the processes)
    bool use_ope;

} NP_RuntimeOptions;

// Load a model given a path and options
void NP_LoadNeuropodWithOpts(const char *             neuropod_path,
                             const NP_RuntimeOptions *options,
                             NP_Neuropod **           model,
                             NP_Status *              status);

// Free a model
void NP_FreeNeuropod(NP_Neuropod *model);

// Run inference
// Note: The caller is responsible for freeing the returned NP_NeuropodValueMap
void NP_Infer(NP_Neuropod *model, const NP_NeuropodValueMap *inputs, NP_NeuropodValueMap **outputs, NP_Status *status);

// Run inference with a set of requested outputs
// requested_outputs should be a null terminated array of char ptr containing the names of requested outputs
// Note: The caller is responsible for freeing the returned NP_NeuropodValueMap
void NP_InferWithRequestedOutputs(NP_Neuropod *              model,
                                  const NP_NeuropodValueMap *inputs,
                                  const char **              requested_outputs,
                                  NP_NeuropodValueMap **     outputs,
                                  NP_Status *                status);

// Get information about the model
const char *NP_GetName(NP_Neuropod *model);
const char *NP_GetPlatform(NP_Neuropod *model);

// Get the number of items in the input spec of a model
size_t NP_GetNumInputs(NP_Neuropod *model);

// Get the number of items in the output spec of a model
size_t NP_GetNumOutputs(NP_Neuropod *model);

// Get an item from the input spec of a model
// Returns nullptr if index is out of range
// Note: The caller is responsible for freeing the returned TensorSpec
NP_TensorSpec *NP_GetInputSpec(NP_Neuropod *model, size_t index);

// Get an item from the output spec of a model
// Returns nullptr if index is out of range
// Note: The caller is responsible for freeing the returned TensorSpec
NP_TensorSpec *NP_GetOutputSpec(NP_Neuropod *model, size_t index);

// Get an allocator for a model
// Note: The caller is responsible for freeing the returned TensorAllocator
NP_TensorAllocator *NP_GetAllocator(NP_Neuropod *model);

#ifdef __cplusplus
}
#endif
