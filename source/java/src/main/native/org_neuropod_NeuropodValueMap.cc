#include "org_neuropod_NeuropodValueMap.h"

#include "jclass_register.h"
#include "utils.h"

#include <neuropod/neuropod.hh>

#include <string>

#include <jni.h>

using namespace neuropod::jni;

JNIEXPORT void JNICALL Java_org_neuropod_NeuropodValueMap_nativeDelete(JNIEnv *, jobject, jlong inputHandle)
{
    auto input = reinterpret_cast<neuropod::NeuropodValueMap *>(inputHandle);
    for (auto &entry : *input)
    {
        entry.second.reset();
    }
    delete input;
}

JNIEXPORT jlong JNICALL Java_org_neuropod_NeuropodValueMap_nativeNew(JNIEnv *, jclass)
{
    auto ret = new neuropod::NeuropodValueMap();
    return reinterpret_cast<long>(ret);
}

JNIEXPORT jlong JNICALL Java_org_neuropod_NeuropodValueMap_nativeGetValue(JNIEnv *env,
                                                                          jclass,
                                                                          jstring key,
                                                                          jlong   nativeHandle)
{
    auto neuropodValueMap = reinterpret_cast<neuropod::NeuropodValueMap *>(nativeHandle);
    auto ret              = (*neuropodValueMap)[neuropod::jni::toString(env, key)];
    // TODO: Clear the ownership by weijiad
    return reinterpret_cast<long>(ret.get());
}

JNIEXPORT jobject JNICALL Java_org_neuropod_NeuropodValueMap_nativeGetKeyList(JNIEnv *env, jclass, jlong nativeHandle)
{
    try
    {
        auto    neuropodValueMap = reinterpret_cast<neuropod::NeuropodValueMap *>(nativeHandle);
        jobject ret              = env->NewObject(java_util_ArrayList, java_util_ArrayList_, neuropodValueMap->size());
        for (const auto &entry : *neuropodValueMap)
        {
            jstring key = env->NewStringUTF(entry.first.c_str());
            env->CallBooleanMethod(ret, java_util_ArrayList_add, key);
            // Normally we do not need to delete local ref manually. But as Java jni only have limited number of local ref
            // support, in a long loop we need to do so.
            env->DeleteLocalRef(key);
        }
        return ret;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}