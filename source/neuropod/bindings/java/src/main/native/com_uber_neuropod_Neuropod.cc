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

#include "com_uber_neuropod_Neuropod.h"

#include "com_uber_neuropod_LibraryLoader.h"
#include "jclass_register.h"
#include "neuropod/core/generic_tensor.hh"
#include "neuropod/neuropod.hh"
#include "neuropod/tests/ope_overrides.hh"
#include "utils.h"

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <jni.h>

using namespace neuropod::jni;

namespace
{
jobject toJavaTensorSpecList(JNIEnv *env, const std::vector<neuropod::TensorSpec> &specs)
{
    jobject ret = env->NewObject(java_util_ArrayList, java_util_ArrayList_, specs.size());
    for (const auto &tensorSpec : specs)
    {
        auto    type = getTensorTypeField(env, tensorTypeToString(tensorSpec.type).c_str());
        jstring name = env->NewStringUTF(tensorSpec.name.c_str());
        jobject dims = env->NewObject(java_util_ArrayList, java_util_ArrayList_, tensorSpec.dims.size());
        for (const auto &dim : tensorSpec.dims)
        {
            // Dim is symbol
            if (dim.value == -2)
            {
                jstring symbol = env->NewStringUTF(dim.symbol.c_str());
                jobject javaDim =
                    env->NewObject(com_uber_neuropod_Dimension, com_uber_neuropod_Dimension_symbol_, symbol);
                env->CallBooleanMethod(dims, java_util_ArrayList_add, javaDim);
                env->DeleteLocalRef(javaDim);
                env->DeleteLocalRef(symbol);
            }
            else
            {
                jobject javaDim =
                    env->NewObject(com_uber_neuropod_Dimension, com_uber_neuropod_Dimension_value_, dim.value);
                env->CallBooleanMethod(dims, java_util_ArrayList_add, javaDim);
                env->DeleteLocalRef(javaDim);
            }
        }
        jobject javaTensorSpec =
            env->NewObject(com_uber_neuropod_TensorSpec, com_uber_neuropod_TensorSpec_, name, type, dims);
        env->CallBooleanMethod(ret, java_util_ArrayList_add, javaTensorSpec);
        env->DeleteLocalRef(name);
        env->DeleteLocalRef(dims);
        env->DeleteLocalRef(type);
    }
    return ret;
}
} // namespace

JNIEXPORT jlong JNICALL Java_com_uber_neuropod_Neuropod_nativeNew__Ljava_lang_String_2J(JNIEnv *env,
                                                                                        jclass,
                                                                                        jstring path,
                                                                                        jlong   optHandle)
{
    try
    {
        neuropod::RuntimeOptions opts;
        if (optHandle != 0)
        {
            opts = *reinterpret_cast<neuropod::RuntimeOptions *>(optHandle);
        }
        auto                convertedPath = toString(env, path);
        neuropod::Neuropod *ret           = nullptr;
        if (isTestMode)
        {
            ret = new neuropod::Neuropod(convertedPath, detail::ope_backend_location_overrides, opts);
        }
        else
        {
            ret = new neuropod::Neuropod(convertedPath, opts);
        }
        return reinterpret_cast<jlong>(ret);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return reinterpret_cast<jlong>(nullptr);
}

JNIEXPORT void JNICALL Java_com_uber_neuropod_Neuropod_nativeDelete(JNIEnv *env, jobject obj, jlong handle)
{
    try
    {

        delete reinterpret_cast<neuropod::Neuropod *>(handle);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
}

JNIEXPORT void JNICALL Java_com_uber_neuropod_Neuropod_nativeLoadModel(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto model = reinterpret_cast<neuropod::Neuropod *>(handle);
        model->load_model();
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
}

JNIEXPORT jstring JNICALL Java_com_uber_neuropod_Neuropod_nativeGetName(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto model = reinterpret_cast<neuropod::Neuropod *>(handle);
        return env->NewStringUTF(model->get_name().c_str());
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}

JNIEXPORT jstring JNICALL Java_com_uber_neuropod_Neuropod_nativeGetPlatform(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto model = reinterpret_cast<neuropod::Neuropod *>(handle);
        return env->NewStringUTF(model->get_platform().c_str());
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}

JNIEXPORT jobject JNICALL Java_com_uber_neuropod_Neuropod_nativeGetInputs(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto model     = reinterpret_cast<neuropod::Neuropod *>(handle);
        auto inputSpec = model->get_inputs();
        return toJavaTensorSpecList(env, inputSpec);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}

JNIEXPORT jobject JNICALL Java_com_uber_neuropod_Neuropod_nativeGetOutputs(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto model   = reinterpret_cast<neuropod::Neuropod *>(handle);
        auto outputs = model->get_outputs();
        return toJavaTensorSpecList(env, outputs);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}

JNIEXPORT jlong JNICALL Java_com_uber_neuropod_Neuropod_nativeGetAllocator(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto model = reinterpret_cast<neuropod::Neuropod *>(handle);
        return reinterpret_cast<jlong>(toHeap(model->get_tensor_allocator()));
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return reinterpret_cast<jlong>(nullptr);
}

JNIEXPORT jlong JNICALL Java_com_uber_neuropod_Neuropod_nativeGetGenericAllocator(JNIEnv *env, jclass)
{
    try
    {
        std::shared_ptr<neuropod::NeuropodTensorAllocator> allocator = neuropod::get_generic_tensor_allocator();
        return reinterpret_cast<jlong>(toHeap(allocator));
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return reinterpret_cast<jlong>(nullptr);
}

JNIEXPORT jobject JNICALL Java_com_uber_neuropod_Neuropod_nativeInfer(
    JNIEnv *env, jclass, jobjectArray entryArray, jobject requestedOutputsJava, jlong modelHandle)
{
    try
    {

        // Prepare requestedOutputs
        std::vector<std::string> requestedOutputs;
        if (requestedOutputsJava != nullptr)
        {
            jsize size = env->CallIntMethod(requestedOutputsJava, java_util_ArrayList_size);
            for (jsize i = 0; i < size; i++)
            {
                jstring element =
                    static_cast<jstring>(env->CallObjectMethod(requestedOutputsJava, java_util_ArrayList_get, i));
                requestedOutputs.emplace_back(toString(env, element));
                env->DeleteLocalRef(element);
            }
        }

        // Fill in NeuropodValueMap
        jsize                      entrySize = env->GetArrayLength(entryArray);
        neuropod::NeuropodValueMap nativeMap;
        for (jsize i = 0; i < entrySize; i++)
        {
            jobject     entry = env->GetObjectArrayElement(entryArray, i);
            std::string key =
                toString(env, static_cast<jstring>(env->CallObjectMethod(entry, java_util_Map_Entry_getKey)));
            jobject value        = env->CallObjectMethod(entry, java_util_Map_Entry_getValue);
            jlong   tensorHandle = env->CallLongMethod(value, com_uber_neuropod_NeuropodTensor_getHandle);
            if (tensorHandle == 0)
            {
                throw std::runtime_error("deallocated object");
            }
            nativeMap[key] = (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(tensorHandle));
            env->DeleteLocalRef(entry);
            env->DeleteLocalRef(value);
        }

        auto model       = reinterpret_cast<neuropod::Neuropod *>(modelHandle);
        auto inferredMap = model->infer(nativeMap, requestedOutputs);

        // Put data to Java Map
        auto ret = env->NewObject(java_util_HashMap, java_util_HashMap_);
        for (auto &entry : *inferredMap)
        {
            jobject javaTensor = env->NewObject(com_uber_neuropod_NeuropodTensor,
                                                com_uber_neuropod_NeuropodTensor_,
                                                reinterpret_cast<jlong>(toHeap((entry.second))));
            env->CallObjectMethod(ret, java_util_HashMap_put, env->NewStringUTF(entry.first.c_str()), javaTensor);
            env->DeleteLocalRef(javaTensor);
        }
        return ret;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}
