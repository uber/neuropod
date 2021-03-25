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

#include "com_uber_neuropod_LibraryLoader.h"

#include "jclass_register.h"
#include "utils.h"
#include "neuropod/internal/logging.hh"

#include <string>
#include <dlfcn.h>

#include <jni.h>

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for Java API methods
JNIEXPORT jboolean JNICALL Java_com_uber_neuropod_LibraryLoader_nativeIsLoaded(JNIEnv * /*unused*/, jclass /*unused*/)
{
    return JNI_TRUE;
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for Java API methods
JNIEXPORT void JNICALL Java_com_uber_neuropod_LibraryLoader_nativeExport(JNIEnv *env,
                                                                         jclass /*unused*/,
                                                                         jstring libPath)
{
    std::string oriPath = getenv("PATH");
    setenv("PATH", (oriPath + ":" + neuropod::jni::to_string(env, libPath)).c_str(), 1 /* Overwrite */);

    std::string libNeuropod("libneuropod.so");
    if (dlopen(libNeuropod.c_str(), RTLD_LAZY | RTLD_GLOBAL) == nullptr)
    {
            const auto err = dlerror();
            if (err == nullptr)
            {
                SPDLOG_TRACE("Loading '{}' failed, but no error message was avaliable", libNeuropod);
            }
            else
            {
                SPDLOG_TRACE("Loading '{}' failed. Error from dlopen: {}", libNeuropod, err);
            }
    }
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for Java API methods
JNIEXPORT jlong JNICALL Java_com_uber_neuropod_LibraryLoader_nativeSetEnv(JNIEnv *env,
                                                                         jclass /*unused*/,
                                                                         jstring name,
                                                                         jstring value)
{
    return setenv(neuropod::jni::to_string(env, name).c_str(), neuropod::jni::to_string(env, value).c_str(), 1 /* Overwrite */);
}
