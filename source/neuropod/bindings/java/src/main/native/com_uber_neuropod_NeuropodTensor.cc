#include "com_uber_neuropod_NeuropodTensor.h"

#include "jclass_register.h"
#include "neuropod/neuropod.hh"
#include "utils.h"

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

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
            throw std::runtime_error("unsupported tensor type: " + tensorTypeToString(tensorType));
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

JNIEXPORT jobject JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeToStringList(JNIEnv *env,
                                                                                   jclass,
                                                                                   jlong nativeHandle)
{
    try
    {
        auto neuropodTensor =
            (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(nativeHandle))->as_tensor();
        auto tensorType = neuropodTensor->get_tensor_type();
        if (tensorType != neuropod::STRING_TENSOR)
        {
            throw std::runtime_error("unexpected tensor type, should be STRING_TENSOR but found:" +
                                     tensorTypeToString(tensorType));
        }

        auto    size = neuropodTensor->get_num_elements();
        jobject ret  = env->NewObject(java_util_ArrayList, java_util_ArrayList_, size);
        if (!ret)
        {
            throw std::runtime_error("out of memory: cannot create ArrayList");
        }

        auto typedTensor = neuropodTensor->as_typed_tensor<std::string>();

        std::function<void(string_accessor_type *)> mapFunc = [env, ret](string_accessor_type *elem) {
            // StringProxy supports conversion to std:string.
            std::string tmpStr(*elem);
            jstring     convertedElem = env->NewStringUTF(tmpStr.c_str());
            env->CallBooleanMethod(ret, java_util_ArrayList_add, convertedElem);
            env->DeleteLocalRef(convertedElem);
        };

        auto dims = typedTensor->get_dims();
        switch (dims.size())
        {
        case 1:
            mapStringTensor(typedTensor->accessor<1>(), mapFunc, dims);
            break;
        case 2:
            mapStringTensor(typedTensor->accessor<2>(), mapFunc, dims);
            break;
        case 3:
            mapStringTensor(typedTensor->accessor<3>(), mapFunc, dims);
            break;
        case 4:
            mapStringTensor(typedTensor->accessor<4>(), mapFunc, dims);
            break;
        default:
            // Here copy data twice
            const auto &elementList = typedTensor->get_data_as_vector();
            for (const auto &elem : elementList)
            {
                jstring convertedElem = env->NewStringUTF(elem.c_str());
                env->CallBooleanMethod(ret, java_util_ArrayList_add, convertedElem);
                env->DeleteLocalRef(convertedElem);
            }
        }
        return ret;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}

JNIEXPORT jstring JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeGetString(JNIEnv *env,
                                                                                jclass,
                                                                                jlong index,
                                                                                jlong handle)
{
    try
    {
        auto stringTensor = (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle))
                                ->as_tensor()
                                ->as_typed_tensor<std::string>();
        const auto &       strList = stringTensor->get_data_as_vector();
        const std::string &elem    = strList[index];
        return env->NewStringUTF(elem.c_str());
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}
