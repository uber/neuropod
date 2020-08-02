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
