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

#pragma once

#include <jni.h>

namespace neuropod
{
namespace jni
{
extern jclass    java_util_ArrayList;
extern jmethodID java_util_ArrayList_;
extern jmethodID java_util_ArrayList_add;
extern jmethodID java_util_ArrayList_get;
extern jmethodID java_util_ArrayList_size;

extern jclass    java_util_HashMap;
extern jmethodID java_util_HashMap_;
extern jmethodID java_util_HashMap_put;

extern jclass    java_util_Map_Entry;
extern jmethodID java_util_Map_Entry_getKey;
extern jmethodID java_util_Map_Entry_getValue;

extern jclass com_uber_neuropod_TensorType;

extern jclass    com_uber_neuropod_TensorSpec;
extern jmethodID com_uber_neuropod_TensorSpec_;

extern jclass    com_uber_neuropod_Dimension;
extern jmethodID com_uber_neuropod_Dimension_value_;
extern jmethodID com_uber_neuropod_Dimension_symbol_;

extern jclass    com_uber_neuropod_NeuropodTensor;
extern jmethodID com_uber_neuropod_NeuropodTensor_;
extern jmethodID com_uber_neuropod_NeuropodTensor_getHandle;

extern jclass com_uber_neuropod_NeuropodJNIException;

extern bool isTestMode;
} // namespace jni
} // namespace neuropod