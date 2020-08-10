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

#include "utils.h"

#include "jclass_register.h"
#include "neuropod/neuropod.hh"

#include <stdexcept>
#include <string>

#include <jni.h>

namespace neuropod
{
namespace jni
{

const std::string TENSOR_TYPE = "Lcom/uber/neuropod/TensorType;";

std::string toString(JNIEnv *env, jstring target)
{
    const char *raw = env->GetStringUTFChars(target, NULL);
    std::string res(raw);
    env->ReleaseStringUTFChars(target, raw);
    return res;
}

jclass findClass(JNIEnv *env, const char *name)
{
    jclass ret = env->FindClass(name);
    if (reinterpret_cast<jlong>(ret) == 0)
    {
        throw std::runtime_error(std::string("Class not found: ") + name);
    }
    return ret;
}

jmethodID getMethodID(JNIEnv *env, jclass clazz, const char *name, const char *sig)
{
    jmethodID ret = env->GetMethodID(clazz, name, sig);
    if (reinterpret_cast<jlong>(ret) == 0)
    {
        throw std::runtime_error(std::string("Method ID not found: ") + name + sig);
    }
    return ret;
}

jobject getTensorTypeField(JNIEnv *env, std::string fieldName)
{
    jfieldID field = env->GetStaticFieldID(com_uber_neuropod_TensorType, fieldName.c_str(), TENSOR_TYPE.c_str());
    if (reinterpret_cast<jlong>(field) == 0)
    {
        throw std::runtime_error(std::string("Field not found: ") + fieldName);
    }
    return env->GetStaticObjectField(com_uber_neuropod_TensorType, field);
}

std::string tensorTypeToString(TensorType type)
{
    std::string       typeString;
    std::stringstream ss;
    ss << type;
    ss >> typeString;
    return typeString;
}

void throwJavaException(JNIEnv *env, const char *message)
{
    env->ThrowNew(com_uber_neuropod_NeuropodJNIException, message);
}

} // namespace jni
} // namespace neuropod
