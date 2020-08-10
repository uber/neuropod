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

#include "com_uber_neuropod_RuntimeOptions_RuntimeOptionsNative.h"

#include "neuropod/neuropod.hh"
#include "utils.h"

#include <exception>
#include <string>

#include <jni.h>

using namespace neuropod::jni;

JNIEXPORT jlong JNICALL
Java_com_uber_neuropod_RuntimeOptions_00024RuntimeOptionsNative_nativeCreate(JNIEnv *env,
                                                                             jclass,
                                                                             jboolean useOpe,
                                                                             jboolean freeMemoryEveryCycle,
                                                                             jstring  jControlQueueName,
                                                                             jint     visibleDevice,
                                                                             jboolean loadModelAtConstruction,
                                                                             jboolean disableShapeAndTypeChecking)
{
    try
    {
        std::string controlQueueName              = toString(env, jControlQueueName);
        auto        opts                          = new neuropod::RuntimeOptions();
        opts->use_ope                             = (useOpe == JNI_TRUE);
        opts->ope_options.free_memory_every_cycle = (freeMemoryEveryCycle == JNI_TRUE);
        opts->ope_options.control_queue_name      = controlQueueName;
        opts->visible_device                      = static_cast<int32_t>(visibleDevice);
        opts->load_model_at_construction          = (loadModelAtConstruction == JNI_TRUE);
        opts->disable_shape_and_type_checking     = (disableShapeAndTypeChecking == JNI_TRUE);
        return reinterpret_cast<jlong>(opts);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return reinterpret_cast<jlong>(nullptr);
}

JNIEXPORT void JNICALL Java_com_uber_neuropod_RuntimeOptions_00024RuntimeOptionsNative_nativeDelete(JNIEnv *env,
                                                                                                    jobject,
                                                                                                    jlong handle)
{
    try
    {
        delete reinterpret_cast<neuropod::RuntimeOptions *>(handle);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
}
