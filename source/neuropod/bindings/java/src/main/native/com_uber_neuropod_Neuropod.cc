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
