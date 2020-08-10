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
jclass    java_util_ArrayList;
jmethodID java_util_ArrayList_;
jmethodID java_util_ArrayList_add;
jmethodID java_util_ArrayList_get;
jmethodID java_util_ArrayList_size;

jclass    com_uber_neuropod_TensorSpec;
jmethodID com_uber_neuropod_TensorSpec_;

jclass    com_uber_neuropod_Dimension;
jmethodID com_uber_neuropod_Dimension_value_;
jmethodID com_uber_neuropod_Dimension_symbol_;

jclass com_uber_neuropod_TensorType;

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
    try
    {
        java_util_ArrayList      = static_cast<jclass>(env->NewGlobalRef(findClass(env, "java/util/ArrayList")));
        java_util_ArrayList_     = getMethodID(env, java_util_ArrayList, "<init>", "(I)V");
        java_util_ArrayList_add  = getMethodID(env, java_util_ArrayList, "add", "(Ljava/lang/Object;)Z");
        java_util_ArrayList_get  = getMethodID(env, java_util_ArrayList, "get", "(I)Ljava/lang/Object;");
        java_util_ArrayList_size = getMethodID(env, java_util_ArrayList, "size", "()I");

        com_uber_neuropod_TensorSpec =
            static_cast<jclass>(env->NewGlobalRef(findClass(env, "com/uber/neuropod/TensorSpec")));
        com_uber_neuropod_TensorSpec_ =
            getMethodID(env,
                        com_uber_neuropod_TensorSpec,
                        "<init>",
                        "(Ljava/lang/String;Lcom/uber/neuropod/TensorType;Ljava/util/List;)V");

        com_uber_neuropod_Dimension =
            static_cast<jclass>(env->NewGlobalRef(findClass(env, "com/uber/neuropod/Dimension")));
        com_uber_neuropod_Dimension_value_ = getMethodID(env, com_uber_neuropod_Dimension, "<init>", "(J)V");
        com_uber_neuropod_Dimension_symbol_ =
            getMethodID(env, com_uber_neuropod_Dimension, "<init>", "(Ljava/lang/String;)V");

        com_uber_neuropod_TensorType =
            static_cast<jclass>(env->NewGlobalRef(findClass(env, "com/uber/neuropod/TensorType")));
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
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
    env->DeleteGlobalRef(java_util_ArrayList);

    env->DeleteGlobalRef(com_uber_neuropod_Dimension);
    env->DeleteGlobalRef(com_uber_neuropod_TensorSpec);
    env->DeleteGlobalRef(com_uber_neuropod_TensorType);
    env->DeleteGlobalRef(com_uber_neuropod_NeuropodJNIException);
}
