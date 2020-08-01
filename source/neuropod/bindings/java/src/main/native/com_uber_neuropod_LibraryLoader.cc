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

#include "utils.h"

#include <string>

#include <jni.h>

JNIEXPORT jboolean JNICALL Java_com_uber_neuropod_LibraryLoader_nativeIsLoaded(JNIEnv *, jclass)
{
    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_com_uber_neuropod_LibraryLoader_nativeExport(JNIEnv *env, jclass, jstring libPath)
{
    std::string oriPath = getenv("PATH");
    setenv("PATH", (oriPath + ":" + neuropod::jni::toString(env, libPath)).c_str(), 1 /* Overwrite */);
}
