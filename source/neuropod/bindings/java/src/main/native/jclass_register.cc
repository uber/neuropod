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

#include "jclass_register.h"

#include "utils.h"

#include <exception>

#include <jni.h>

namespace neuropod
{
namespace jni
{

jclass com_uber_neuropod_NeuropodJNIException;

jint JNI_VERSION = JNI_VERSION_1_8;

bool isTestMode = false;
} // namespace jni
} // namespace neuropod

using namespace neuropod::jni;

// This function is called when the JNI is loaded.
jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv *env;

    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION) != JNI_OK)
    {
        return JNI_ERR;
    }
    // Move this exception class out of try catch block to avoid unexpected error when throw a java exception and the
    // exception type is wrong
    com_uber_neuropod_NeuropodJNIException =
        static_cast<jclass>(env->NewGlobalRef(findClass(env, "com/uber/neuropod/NeuropodJNIException")));
    // Return the JNI Version as required by method
    return JNI_VERSION;
}

// This function is called when the JNI is unloaded.
void JNI_OnUnload(JavaVM *vm, void *reserved)
{
    // Obtain the JNIEnv from the VM
    JNIEnv *env;
    vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION);
    // Destroy the global references
    env->DeleteGlobalRef(com_uber_neuropod_NeuropodJNIException);
}
