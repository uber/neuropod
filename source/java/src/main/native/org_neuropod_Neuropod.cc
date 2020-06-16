#include "org_neuropod_Neuropod.h"

#include "jclass_register.h"
#include "utils.h"

#include <neuropod/neuropod.hh>

#include <exception>
#include <memory>
#include <sstream>
#include <string>

#include <jni.h>

using namespace neuropod::jni;

JNIEXPORT jlong JNICALL Java_org_neuropod_Neuropod_nativeNew__Ljava_lang_String_2(JNIEnv *env, jclass, jstring path)
{
    neuropod::RuntimeOptions opts;
    opts.use_ope = true;
    try
    {
        auto                convertedPath = toString(env, path);
        neuropod::Neuropod *ret           = new neuropod::Neuropod(convertedPath, opts);
        return reinterpret_cast<jlong>(ret);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return reinterpret_cast<jlong>(nullptr);
}

JNIEXPORT jlong JNICALL Java_org_neuropod_Neuropod_nativeNew__Ljava_lang_String_2J(JNIEnv *env,
                                                                                   jclass,
                                                                                   jstring path,
                                                                                   jlong   optHandle)
{
    auto opts = reinterpret_cast<neuropod::RuntimeOptions *>(optHandle);
    try
    {
        auto                convertedPath = toString(env, path);
        neuropod::Neuropod *ret           = new neuropod::Neuropod(convertedPath, *opts);
        return reinterpret_cast<jlong>(ret);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return reinterpret_cast<jlong>(nullptr);
}

JNIEXPORT jlong JNICALL Java_org_neuropod_Neuropod_nativeInfer(JNIEnv *env,
                                                               jclass,
                                                               jlong inputHandle,
                                                               jlong modelHandle)
{
    auto input = reinterpret_cast<neuropod::NeuropodValueMap *>(inputHandle);
    auto model = reinterpret_cast<neuropod::Neuropod *>(modelHandle);
    try
    {
        auto ret = model->infer(*input).release();
        return reinterpret_cast<jlong>(ret);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return reinterpret_cast<jlong>(nullptr);
}

JNIEXPORT jlong JNICALL Java_org_neuropod_Neuropod_nativeCreateTensorsFromMemory(JNIEnv *env,
                                                                                 jclass,
                                                                                 jobject inputs,
                                                                                 jlong   modelHandle)
{
    auto ret = new neuropod::NeuropodValueMap();
    try
    {
        auto model     = reinterpret_cast<neuropod::Neuropod *>(modelHandle);
        auto inputSpec = model->get_inputs();
        auto allocator = model->get_tensor_allocator();

        for (const auto &tensorSpec : inputSpec)
        {
            jstring key = env->NewStringUTF(tensorSpec.name.c_str());
            (*ret)[tensorSpec.name] =
                neuropod::jni::createTesnorFromJavaMemory(allocator,
                                                          env,
                                                          env->CallObjectMethod(inputs, java_util_Map_get, key),
                                                          tensorSpec.type,
                                                          neuropod::jni::getDefaultInputDim(tensorSpec));
        }
        return reinterpret_cast<jlong>(ret);
    }
    catch (const std::exception &e)
    {
        delete ret;
        throwJavaException(env, e.what());
    }
    return reinterpret_cast<jlong>(nullptr);
}

JNIEXPORT void JNICALL Java_org_neuropod_Neuropod_nativeDelete(JNIEnv *, jobject obj, jlong handle)
{
    auto pointer = reinterpret_cast<neuropod::Neuropod *>(handle);
    delete pointer;
}

JNIEXPORT jobject JNICALL Java_org_neuropod_Neuropod_nativeGetInputFeatureKeys(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto model     = reinterpret_cast<neuropod::Neuropod *>(handle);
        auto inputSpec = model->get_inputs();
        jobject ret = env->NewObject(java_util_ArrayList, java_util_ArrayList_, inputSpec.size());
        for (const auto &tensorSpec : inputSpec)
        {
            jstring key = env->NewStringUTF(tensorSpec.name.c_str());
            env->CallBooleanMethod(ret, java_util_ArrayList_add, key);
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

JNIEXPORT jobject JNICALL Java_org_neuropod_Neuropod_nativeGetInputFeatureDataTypes(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto model     = reinterpret_cast<neuropod::Neuropod *>(handle);
        auto inputSpec = model->get_inputs();
        jobject ret = env->NewObject(java_util_ArrayList, java_util_ArrayList_, inputSpec.size());
        for (const auto &tensorSpec : inputSpec)
        {

            env->CallBooleanMethod(ret,
                                   java_util_ArrayList_add,
                                   getFieldObject(env, org_neuropod_DataType, tensorTypeToString(tensorSpec.type).c_str()));
        }
        return ret;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}