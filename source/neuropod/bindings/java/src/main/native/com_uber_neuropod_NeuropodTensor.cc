#include "com_uber_neuropod_NeuropodTensor.h"

#include "jclass_register.h"
#include "neuropod/neuropod.hh"
#include "utils.h"

#include <exception>
#include <memory>

#include <jni.h>

using namespace neuropod::jni;

JNIEXPORT void JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeDoDelete(JNIEnv *env, jobject, jlong handle)
{
    try
    {
        // Java NeuropodTensor stores a pointer to a shared_ptr
        auto tensorPtr = reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle);
        std::unique_ptr<std::shared_ptr<neuropod::NeuropodValue>> scopeHolder(tensorPtr);
        tensorPtr->reset();
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
}

JNIEXPORT jobject JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeGetBuffer(JNIEnv *env, jclass, jlong nativeHandle)
{
    try
    {
        auto neuropodTensor =
            (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(nativeHandle))->as_tensor();
        auto tensorType = neuropodTensor->get_tensor_type();
        switch (tensorType)
        {
        case neuropod::FLOAT_TENSOR: {
            return createDirectBuffer<float>(env, neuropodTensor);
        }
        case neuropod::DOUBLE_TENSOR: {
            return createDirectBuffer<double>(env, neuropodTensor);
        }
        case neuropod::INT32_TENSOR: {
            return createDirectBuffer<int32_t>(env, neuropodTensor);
        }
        case neuropod::INT64_TENSOR: {
            return createDirectBuffer<int64_t>(env, neuropodTensor);
        }
        default:
            throw std::runtime_error(std::string("Unsupported tensor type: ") + tensorTypeToString(tensorType));
        }
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}

JNIEXPORT jlongArray JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeGetDims(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto       tensor = (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle))->as_tensor();
        auto       dims   = tensor->as_tensor()->get_dims();
        jlongArray result = env->NewLongArray(dims.size());
        if (!result)
        {
            throw std::runtime_error("out of memory");
        }
        env->SetLongArrayRegion(result, 0, dims.size(), reinterpret_cast<jlong *>(dims.data()));
        return result;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}

JNIEXPORT jobject JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeGetTensorType(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto tensor = (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle))->as_tensor();
        auto type   = tensor->as_tensor()->get_tensor_type();
        return getTensorTypeField(env, tensorTypeToString(type).c_str());
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}

JNIEXPORT jlong JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeGetNumberOfElements(JNIEnv *env,
                                                                                        jclass,
                                                                                        jlong handle)
{
    try
    {
        auto tensor = (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle))->as_tensor();
        return tensor->get_num_elements();
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return 0;
}
