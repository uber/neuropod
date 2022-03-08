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

// Inspired by the TensorFlow C API

#pragma once

#include "neuropod/bindings/c/np_tensor.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A collection of tensors. This is the input and output type of `infer`
typedef struct NP_NeuropodValueMap NP_NeuropodValueMap;

// Create a new NeuropodValueMap
NP_NeuropodValueMap *NP_NewValueMap();

// Free a NeuropodValueMap
void NP_FreeValueMap(NP_NeuropodValueMap *nvm);

// Insert a tensor into the provided map
// Overwrites any existing tensors with the same name
// Note: this does not transfer ownership and the caller is still responsible for calling NP_FreeTensor on
// `tensor`
void NP_InsertTensor(NP_NeuropodValueMap *nvm, const char *name, NP_NeuropodTensor *tensor);

// Given a NP_NeuropodValueMap, find a tensor with name `name` and return it
// If such a tensor is not found, returns nullptr
// Note: the caller is responsible for calling NP_FreeTensor on the returned tensor
NP_NeuropodTensor *NP_GetTensor(const NP_NeuropodValueMap *nvm, const char *name);

// Removes a specified tensor from the provided map (if it exists)
void NP_RemoveTensor(NP_NeuropodValueMap *nvm, const char *name);

// A list of tensor names
typedef struct NP_TensorNameList
{
    // The number of items in the array below
    size_t num_tensors;

    // An array of tensor names
    const char *tensor_names[];
} NP_TensorNameList;

// Returns the names of all the tensors stored in a given NP_NeuropodValueMap
// Note: the caller is responsible for calling NP_FreeTensorNameList on the returned tensor
NP_TensorNameList *NP_GetTensorNames(const NP_NeuropodValueMap *nvm);

// Free a NP_TensorNameList
void NP_FreeTensorNameList(NP_TensorNameList *tensor_names);

#ifdef __cplusplus
}
#endif
